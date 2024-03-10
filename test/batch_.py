import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_batch_llama import LlamaForCausalLM
from models.batch_cache import BatchSimpleCache
from utils.batch_decoding import Baseline
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
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

######## model initialization ########
if args.target == 'llama-7B':
    target = LlamaForCausalLM.from_pretrained("/home/hanshis/workspace/NNSPD/models/7B", torch_dtype=torch.float16, device_map="cuda:0")
elif args.target == 'llama-7B-32K':
    target = LlamaForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", torch_dtype=torch.float16, device_map="cuda:0")
elif args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="cuda:0")
elif args.target == 'llama-7B-1M':
    target = LlamaForCausalLM.from_pretrained("LargeWorldModel/LWM-Text-1M", torch_dtype=torch.float16, device_map="cuda:0")
else:
    raise NotImplementedError
target = target.eval()
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)


######## data initialization ########
bsz = args.bsz
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)
tokenized_prompts = [torch.cat(tokenized_prompts[i:i+bsz], dim=0) for i in range(0, len(tokenized_prompts), bsz)]

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

print_config(target, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="Batch Spec (Retrieval)", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

####### cache init #######
draft_cache_budget = args.draft_cache_budget
recent_size = draft_cache_budget - 16 - gamma

cache = BatchSimpleCache(target, prefill+gen_len+16, bsz=bsz)
graph_cache = None

graph_engine = GraphInferenceEngine(target, cache)
# graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)
cache.reset()
cache.print_status()
# graph_cache.print_status()
# draft_cache.print_status()
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

graph_engine.engine.kv_cache.seq_len = torch.full((bsz,), prefill, dtype=torch.int32).cuda()

with torch.inference_mode():
    sentence = torch.randint(low=3, high=30000, size=(bsz, 1)).cuda()
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        graph_engine.inference(sentence)
        graph_engine.engine.kv_cache.seq_len -= 1
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time *10, 1, prefill, 100, "warm up done")


LEN = [1]

with torch.no_grad():
    for l in LEN:
        sentence = torch.randint(low=3, high=30000, size=(bsz, l)).cuda()
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(100):
            graph_engine.inference(sentence)
            graph_engine.engine.kv_cache.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        foward_time = (t2 - t1) / 100 * 1000

        logits = torch.randn(bsz, l, 32000).half().cuda()
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(100):
            next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        torch.cuda.synchronize()
        t2 = time.time()
        sampling_time = (t2 - t1) / 100 * 1000

        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(256):
            logits = graph_engine.inference(next_token)
            next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        torch.cuda.synchronize()
        t2 = time.time()
        all_time = (t2 - t1) / 256 * 1000

        print(f"bsz={bsz}, verify_len={l}, data_len={prefill}, foward_time={foward_time}, sampling_time={sampling_time}, all_time={all_time}")


with torch.inference_mode():
    graph_engine.engine.kv_cache.seq_len = torch.full((bsz,), prefill, dtype=torch.int32).cuda()
    next_token= torch.randint(low=3, high=30000, size=(bsz, 1)).cuda()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(256):
        logits = graph_engine.inference(next_token)
        # next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
    torch.cuda.synchronize()
    t2 = time.time()
    all_time = (t2 - t1) / 256 * 1000

print(f"all_time={all_time}")

######## Warm up for baseline ########
n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Baseline Warmup"):
    Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)

with torch.inference_mode():
    graph_engine.engine.kv_cache.seq_len = torch.full((bsz,), prefill, dtype=torch.int32).cuda()
    next_token= torch.randint(low=3, high=30000, size=(bsz, 1)).cuda()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(256):
        logits = graph_engine.inference(next_token)
        # next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
    torch.cuda.synchronize()
    t2 = time.time()
    all_time = (t2 - t1) / 256 * 1000

print(f"all_time={all_time}")

with torch.no_grad():
    all_latency = []
    for input_ids in tqdm(tokenized_prompts[:1], desc="Baseline Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]
        latency = Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)
        all_latency.append(latency)

    baseline_latency = (sum(all_latency) / len(all_latency)) * 1000
    print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} ms", "red"))

# ######## Warm up for our method ########
# n_warmups = 6
# input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
# for i in tqdm(range(n_warmups), desc="Graph Chain Spec Warmup"):
#     Graph_Chain_Retrieval_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, spec_args={'budget': args.budget, 'draft': args.draft, 'chunk_size': chunk_size, 'baseline': baseline_latency/1000})

# all_acceptance_rate = []
# all_speed = []
# for input_ids in tqdm(tokenized_prompts, desc="Graph Chain Spec Test"):
#     input_ids = input_ids.to(target.device)[:,:prefill]

#     acceptance_rate, speed = Graph_Chain_Retrieval_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=file_path, dataset=args.dataset, spec_args={'budget': args.budget, 'draft': args.draft, 'chunk_size': chunk_size, 'gamma': gamma, 'temperature': temperature, 'top_p': top_p, 'baseline': baseline_latency/1000})
#     all_acceptance_rate.append(acceptance_rate)
#     all_speed.append(speed)

# method_latency = 1000/(sum(all_speed) / len(all_speed))
# print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
# print(colored(f"[Ours-Chain_Retrieval] average latency: {method_latency} ms", "red"))
# print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))