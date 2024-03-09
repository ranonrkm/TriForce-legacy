import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache_utils import FlashSimpleCache, GraphFlashStreamEvictionCache, GraphFlashStreamLLMVerificationCache
from utils.decoding import Graph_Chain_Spec, Baseline
from utils.misc import print_config
from utils.chain_graph_infer import GraphInferenceEngine

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

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
    temperature = 0.6

prefill = args.prefill
gen_len = args.gen_len
gamma = args.gamma
verbose = args.verbose

import socket
host = socket.gethostname()
if 'lovelace' in host:
    file_path = "/home/hanshis/workspace/LongContextInfer/test/report/L40_Ablation_graph_chain_streamllm.csv"
else:
    file_path = "/data/home/beidic/hanshi/LongContextInfer/test/report/A100_Ablation_graph_chain_streamllm.csv"

# print_config(target, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="StreamLLM", spec_args={'budget': args.budget}, dataset=args.dataset)

start_size = 20
recent_size = 4096 - start_size

# draft = LlamaForCausalLM_68M.from_pretrained("/home/hanshis/workspace/LongContextInfer/archive/ckpts/512/step_125", torch_dtype=torch.float16, device_map="auto")

draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="auto")

####### cache init #######
cache = FlashSimpleCache(target, prefill+gen_len+16)
graph_cache = GraphFlashStreamLLMVerificationCache(target, max_budget=start_size+recent_size, prefill=prefill, gamma=gamma, start_size=start_size)
draft_cache = GraphFlashStreamEvictionCache(draft, start_size=16, recent_size=250-16, gamma=gamma)

graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
graph_engine.initialize_cuda_graph(gamma)

cache.print_status()
graph_cache.print_status()
draft_cache.print_status()

all_acceptance_rate = []
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

####### Warm up for baseline ########
n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Baseline Warmup"):
    Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)

all_speed = []
for input_ids in tqdm(tokenized_prompts[:1], desc="Baseline Test"):
    input_ids = input_ids.to(target.device)[:,:prefill]
    speed = Baseline(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)
    all_speed.append(speed)

baseline_latency = 1000/(sum(all_speed) / len(all_speed))
print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} ms", "red"))

n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Graph Chain Spec Warmup"):
    Graph_Chain_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, spec_args={'budget': 4096, 'baseline': baseline_latency/1000})


all_acceptance_rate = []
all_speed = []
for input_ids in tokenized_prompts:
    input_ids = input_ids.to(target.device)[:,:prefill]

    acceptance_rate, speed = Graph_Chain_Spec(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=file_path, dataset=args.dataset, spec_args={'budget': 4096, 'baseline': baseline_latency/1000})
    all_acceptance_rate.append(acceptance_rate)
    all_acceptance_rate.append(acceptance_rate)
    all_speed.append(speed)

method_latency = 1000/(sum(all_speed) / len(all_speed))
print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
print(colored(f"[Ours-Chain_Retrieval] average latency: {method_latency} ms", "red"))
print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))