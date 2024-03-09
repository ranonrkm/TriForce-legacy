# CUDA_VISIBLE_DEVICES=1 python test/e2e_tree.py --prefill 130752 --budget 0.1
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
import time
import gc
import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_llama_tree import LlamaForCausalLM
from models.cache_utils import TREEChunkTopKCache, TREESimpleCache, OffloadingTREESimpleCache
from utils.decoding import TreeBaseline
from utils.misc import print_config, setup_seed, spec_stream
from utils.tree_infer import GraphInferenceEngine, get_sampling_logits, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement
import numpy as np
from utils.SpecTree import SpecTree

import socket
host = socket.gethostname()
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')
    parser.add_argument('--log_csv', action='store_true', help='log_csv')

    parser.add_argument('--tree_size', type=str, default='128')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--budget', type=float, default='0.1')
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    args = parser.parse_args()
    
    return args

args = parse_arguments()
# setup_seed(args.seed)

######## model initialization ########
if args.target == 'llama-7B':
    target = LlamaForCausalLM.from_pretrained("/home/hanshis/workspace/NNSPD/models/7B", torch_dtype=torch.float16, device_map="auto")
elif args.target == 'llama-7B-32K':
    target = LlamaForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", torch_dtype=torch.float16, device_map="auto")
elif args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
elif args.target == 'llama-7B-1M':
    target = LlamaForCausalLM.from_pretrained("LargeWorldModel/LWM-Text-1M", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

target = target.eval()
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)

######## sampling parameters ########

if args.greedy:
    top_k = 1
    top_p = 1
    temperature = 1
else:
    top_k = -1
    top_p = 0.9
    temperature = args.temp

prefill = args.prefill
gen_len = args.gen_len
gamma = args.gamma
verbose = args.verbose

if args.log_csv:
    if 'lovelace' in host:
        file_path = "/home/hanshis/workspace/LongContextInfer/test/report/L40_E2E_graph_chain_retrieval.csv"
    else:
        file_path = "/data/home/beidic/hanshi/LongContextInfer/test/report/A100_E2E_graph_chain_retrieval.csv"
else:
    file_path = None

if 'lovelace' in host:
    align_ckpt = "/home/hanshis/workspace/LongContextInfer/archive/ckpts/512/step_125"
else:
    align_ckpt = "/fsx-storygen/beidic/hanshi/ckpts/Base-128K-256/step_11696"

chunk_size = args.chunk_size
max_budget = int(args.budget * prefill) // chunk_size * chunk_size

print_config(target, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="Tree Graph Chain Spec (Retrieval)", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

####### cache init #######
residual_graph = cuda_graph_for_residual()
path = f'tree/{args.tree_size}.pt'

grow_map = torch.load(path)
tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])


cache = OffloadingTREESimpleCache(target, prefill+gen_len+tree_size+16)
graph_cache = TREEChunkTopKCache(target, max_budget=max_budget, prefill=prefill, tree_size=tree_size, chunk_size=chunk_size)
graph_engine = GraphInferenceEngine(target, cache, graph_cache)

###### Warm up for TreeBaseline ########
n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="TreeBaseline Warmup"):
    TreeBaseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)

all_speed = []
for input_ids in tqdm(tokenized_prompts[:1], desc="TreeBaseline Test"):
    input_ids = input_ids.to(target.device)[:,:prefill]
    speed = TreeBaseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)
    all_speed.append(speed)
TreeBaseline_latency = 1/(sum(all_speed) / len(all_speed))
print(colored(f"[TreeBaseline-Autoregressive] average latency: {TreeBaseline_latency} s", "red"))


sampling_callables = {}
sample_gather_indices = {}
for i in range(draft_step - 1):
    idx_len = len(idx_lists[i])
    num_samples = max(branch_lists[i])
    sampling_callables[i] = cuda_graph_for_sampling_without_replacement(
        max_length=0, idx_len=idx_len, num_samples=num_samples,
        temperature=args.temp, tree_size=tree_size)

for i in range(draft_step - 1):
    ith_gather_list = []
    max_num_samples = max(branch_lists[i])
    for j, branch in enumerate(branch_lists[i]):
        branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
        branch_index = branch_index + j * max_num_samples
        ith_gather_list.append(branch_index)
    ith_gather_list = torch.cat(ith_gather_list)
    sample_gather_indices[i] = ith_gather_list

cache.print_status()
graph_cache.print_status()
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

######## Our method ########
all_speed_up = []
all_acc_list = []
for input_ids in tokenized_prompts:
    input_ids = input_ids[0,:args.prefill].to(target.device)
    dtype = torch.float16
    max_length = prefill+gen_len
    spectree = SpecTree(engine=graph_engine, device='cuda:0', temperature=args.temp, top_p=top_p,
                        max_length=prefill+gen_len, grow_map=grow_map,
                        residual_graph = residual_graph,
                        sampling_callables=sampling_callables,
                        sample_gather_indices = sample_gather_indices,
                        tokenizer=tokenizer)


    with torch.inference_mode():
        n=0
        next_token = spectree.prefill(prefix=input_ids)
        acc_count_list = []

        time1 = time.time()
        while n < gen_len:
            spectree.construct_grow_map(next_token=next_token)
            next_token, acc_count = spectree.verify()
            if next_token is None:
                break
            next_token = next_token.unsqueeze(0)
            n += acc_count
            acc_count_list.append(acc_count)
        time2 = time.time()
        method_latency = (time2 - time1)/n
        print(f"[Avg Accepted Tokens]: {np.array(acc_count_list).mean()}")
        # TreeBaseline_latency = 3.829643356800079
        print(colored(f"[Ours-Chain_Retrieval] average latency: {method_latency} s", "red"))
        print(colored(f"[E2E Speedup]: {TreeBaseline_latency / method_latency}", "red"))

    all_acc_list.append(np.array(acc_count_list).mean())
    all_speed_up.append(TreeBaseline_latency / method_latency)
print(f"[Overall Speedup]: {np.array(all_speed_up).mean()}")
print(f"[Overall Avg Accepted Tokens]: {np.array(all_acc_list).mean()}")