import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
from termcolor import colored
from utils.batch_decoding import Baseline_Dist, Retrieval_Spec_Dist
from models.TP_llama import distributed_init, DistributedLlama
from models.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)
model_name_or_path = "NousResearch/Yarn-Llama-2-7b-128k"

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    parser.add_argument('--budget', type=int,  default=4096)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--file', type=str, default='')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

prefill = args.prefill
bsz = args.bsz
gen_len = args.gen_len
temperature = 0.6
top_p = 0.9
retrieval_budget = args.budget

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, legacy=False)
llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size, prefill=prefill, bsz=bsz, gen_len=gen_len, temperature=temperature, top_p=top_p, flash_attn=True, retrieval_budget=retrieval_budget)
for rank in range(world_size):
    if local_rank == rank:
        print(f"Rank {rank+1}/{world_size} (Device {device}) is initializing parameters")
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()


from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='benchmark', tokenizer=tokenizer, datalen=32768)
input_ids = tokenized_prompts[0][:,:prefill].repeat(bsz, 1).to(device)

# prompts = "Speculative decoding is a technique that allows the model to generate multiple tokens in parallel."
# input_ids = tokenizer.encode(prompts, return_tensors="pt").to(device)
# input_ids = input_ids.repeat(2, 1)

baseline_latency, gen_tokens = Baseline_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
if local_rank == 0:
    # llm.kv_cache.print_status()
    # print(tokenizer.batch_decode(gen_tokens))
    print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} ms", "red"))
dist.barrier()
if args.file:
    with open(args.file, 'a') as f:
        f.write(f"{bsz},{prefill},{baseline_latency},{gen_len}\n")

# retrieval_latency, gen_tokens = Retrieval_Spec_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
# if local_rank == 0:
#     print(tokenizer.batch_decode(gen_tokens))
#     llm.kv_cache.print_status()
#     print(colored(f"[Retrieval-Speculative-Decoding] average latency: {retrieval_latency} ms", "red"))
# dist.barrier()