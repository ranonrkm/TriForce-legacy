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
import numpy as np
import time

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
    parser.add_argument('--T', type=int, default=1000, help='repeat times')
    parser.add_argument('--offloading', action='store_true', help='offloading')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

prefill = args.prefill
bsz = args.bsz
gen_len = args.gen_len
temperature = 0.6
top_p = 0.9
retrieval_budget = args.budget
gamma = args.gamma
T = args.T

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, legacy=False)
llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size, prefill=prefill, bsz=bsz, gen_len=gen_len, temperature=temperature, top_p=top_p, flash_attn=True, retrieval_budget=retrieval_budget, gamma=gamma, kv_offload=args.offloading)
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

llm.prefill(input_ids=input_ids[:,:-1])
logits = llm.build_retrieval_cache(input_ids=input_ids[:,-1:])

LEN = [1,2,4,5,6,7,8,9,10,11,12,12,13,14,15,16,32,64,96,128]

with torch.inference_mode():
    for l in LEN:
        sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(llm.device)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
            llm.inference(input_ids=sentence)
            llm.kv_cache.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        foward_time = (t2 - t1) / T * 1000
        assert llm.kv_cache.seq_len.min() == prefill
        assert llm.kv_cache.seq_len.max() == prefill

        torch.cuda.synchronize()
        t1 = time.time()
        position_ids = llm.kv_cache.seq_len[:, None] + gamma -1
        input_ids = torch.randint(low=3, high=30000, size=(bsz, 1)).to(llm.device)
        for _ in range(T):
            llm.retrieval_inference(input_ids=input_ids, gamma_offset=gamma -1, position_ids=position_ids)
        torch.cuda.synchronize()
        t2 = time.time()
        draft_time = (t2 - t1) / T * 1000

        if local_rank == 0:
            print(f"bsz={bsz}, prefill={prefill}, verify_len={l}, foward_time={foward_time}, draft_time={draft_time}", flush=True)

