import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_batch_llama import LlamaForCausalLM
from models.batch_cache import BatchSimpleCache, BatchRetrievalCache
from utils.batch_decoding import Baseline, Retrieval_Spec
from utils.misc import print_config
from utils.batch_infer import GraphInferenceEngine
import socket
host = socket.gethostname()
import argparse
import time
from utils.sampling import sample, norm_logits, max_fn

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M-align', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--bsz', type=int, default=2, help='bsz')
    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

######## model initialization ########
if args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)
elif args.target == 'llama-7B-1M':
    target = LlamaForCausalLM.from_pretrained("LargeWorldModel/LWM-Text-1M", torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("LargeWorldModel/LWM-Text-1M", use_fast=True, legacy=False)
elif args.target == 'Yi-6B-200K':
    target = LlamaForCausalLM.from_pretrained("01-ai/Yi-6B-200K", torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-200K", use_fast=True, legacy=False)
else:
    raise NotImplementedError
target = target.eval()


######## data initialization ########
bsz = args.bsz
from data.dataset import get_dataset
ds_tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)
tokenized_prompts = []
doc_id = 0
i = 0
for _ in range(10):
    cur_batch = []
    for _ in range(args.bsz):
        if i + args.prefill > ds_tokenized_prompts[doc_id].size(1):
            i = 0
            doc_id += 1
        assert doc_id < len(ds_tokenized_prompts)
        cur_batch.append(ds_tokenized_prompts[doc_id][:, i : i + args.prefill])
        i += args.prefill
    cur_batch = torch.cat(cur_batch, dim=0)
    tokenized_prompts.append(cur_batch)

# tokenized_prompts = [torch.cat(tokenized_prompts[i:i+bsz], dim=0) for i in range(0, len(tokenized_prompts), bsz)]

######## sampling parameters ########
top_k = -1
top_p = args.top_p
temperature = args.temp

prefill = args.prefill
gen_len = args.gen_len
gamma = args.gamma
verbose = args.verbose


if 'lovelace' in host:
    file_path = "/home/hanshis/workspace/LongContextInfer/test/report/L40_Ablation_graph_chain_retrieval.csv"
else:
    file_path = "/data/home/beidic/hanshi/LongContextInfer/test/report/A100_Ablation_graph_chain_retrieval.csv"

if 'lovelace' in host:
    align_ckpt = "/home/hanshis/workspace/LongContextInfer/archive/ckpts/512/step_125"
else:
    align_ckpt = "/fsx-storygen/beidic/hanshi/ckpts/Base-128K-256/step_11696"

chunk_size = args.chunk_size
max_budget = args.budget
file_path = None
print_config(target, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="Batch Spec (Retrieval)", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

####### cache init #######
cache = BatchSimpleCache(target, int(prefill+gen_len*2), bsz=bsz)
graph_cache = BatchRetrievalCache(target, max_budget, bsz=bsz, prefill=prefill, chunk_size=chunk_size, gamma=gamma)

graph_engine = GraphInferenceEngine(target, cache, graph_cache=graph_cache, bsz=bsz)
graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)
cache.print_status()
# graph_cache.print_status()
# draft_cache.print_status()
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

####### Warm up for baseline ########
n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Baseline Warmup"):
    Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False)

with torch.no_grad():
    all_latency = []
    for input_ids in tqdm(tokenized_prompts, desc="Baseline Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]
        # print(tokenizer.batch_decode(input_ids))
        latency, gen_tokens = Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=False)
        all_latency.append(latency)
        # print(tokenizer.batch_decode(gen_tokens))
        # exit()

    baseline_latency = (sum(all_latency) / len(all_latency)) * 1000
    print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} ms", "red"))

# DEBUG: inspect the generated tokens quality
# print(tokenizer.batch_decode(gen_tokens))

######### Warm up for our method ########
n_warmups = 3
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Graph Chain Spec Warmup"):
    acceptance_rate, avg_tokens, latency, gen_tokens = Retrieval_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=True, file_path=None, dataset=args.dataset)
print(tokenizer.batch_decode(gen_tokens))
all_acceptance_rate = []
all_latency = []

for input_ids in tqdm(tokenized_prompts, desc="Graph Chain Spec Test"):
    input_ids = input_ids.to(target.device)[:,:prefill]
    if input_ids.size(0) != bsz:
        break
    acceptance_rate, avg_tokens, latency, _ = Retrieval_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset)
    all_acceptance_rate.append(acceptance_rate)
    all_latency.append(latency)

method_latency = (sum(all_latency) / len(all_latency))
print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
print(colored(f"[Ours-Chain_Retrieval] average latency: {method_latency} ms", "red"))
print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))