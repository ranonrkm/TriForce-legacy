import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored

from models.modeling_llama_torch import LlamaForCausalLM
from models.cache_utils import ChunkCache
from utils.decoding import KV_Spec_cache
from utils.misc import print_config, spec_stream

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=1, help='gamma')
    parser.add_argument('--log_csv', action='store_true', help='log_csv')

    parser.add_argument('--chunk_size', type=int, default=16, help='chunk_size')
    parser.add_argument('--budget', type=float, default=0.1, help='budget')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

######## model initialization ########

if args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)

target = target.eval()
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

if args.log_csv:
    import socket
    host = socket.gethostname()
    if 'lovelace' in host:
        file_path = f"/home/hanshis/workspace/LongContextInfer/test/report/L40_retrieval.csv"
    else:
        file_path = "/fsx-storygen/beidic/hanshi/LongContextInfer/test/report/A100_streamllm.csv"
else:
    file_path = None

print_config(target, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="Retrieval KV", spec_args={'chunk_size': args.chunk_size, 'budget': args.budget}, dataset=args.dataset)

######## cache initialization ########
chunk_size = args.chunk_size
budget = args.budget
print(colored(f"chunk_size: {chunk_size}, budget: {budget}", "green"))
target_cache = ChunkCache(target, max_budget=prefill+gen_len+16, chunk_size=chunk_size, budget=budget, skip_start_layers=1, prefill=prefill)

target_cache.print_status()
all_acceptance_rate = []
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

for input_ids in tokenized_prompts:
    input_ids = input_ids.to(target.device)[:,:prefill]

    acceptance_rate = KV_Spec_cache(tokenizer, target, target_cache, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=file_path, dataset=args.dataset, spec_args={'chunk_size': chunk_size, 'budget': budget})
    all_acceptance_rate.append(acceptance_rate)

print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))