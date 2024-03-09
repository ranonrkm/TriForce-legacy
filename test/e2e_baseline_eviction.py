import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

print(f"added {root_dir} to sys.path")

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_llama_torch import LlamaForCausalLM
from models.modeling_llama_68m_v2 import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache_utils import SimpleCache, EvictStreamLLMCache, StreamLLMCache
from utils.decoding import Evict_Spec_cache
from utils.misc import print_config
from utils.sampling import max_fn, sample, norm_logits

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68m-256', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

######## model initialization ########
draft = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="auto")

if args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
else:
    target = LlamaForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16, device_map="auto")

draft = draft.eval()
target = target.eval()

tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m", legacy=False, use_fast=True)

######## sampling parameters ########

if args.greedy:
    top_k = 1
    top_p = 1
    temperature = 1
else:
    top_k = -1
    top_p = 0.9
    temperature = args.temp

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)

prefill = args.prefill
gen_len = args.gen_len
gamma = args.gamma
verbose = args.verbose

import socket
host = socket.gethostname()
if 'lovelace' in host:
    file_path = "/home/hanshis/workspace/LongContextInfer/test/report/L40_Ablation_baseline_evict_streamllm.csv"
elif 'cr-a100-80-0004' in host:
    file_path = "/var/cr06_data/beidic/LongContextInfer/test/report/A100_fake_Ablation_baseline_evict_streamllm.csv"
else:
    file_path = "/data/home/beidic/hanshi/LongContextInfer/test/report/A100_real_Ablation_baseline_evict_streamllm.csv"


print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path, method="Evict StreamLLM", spec_args={'start_size': 16, 'recent_size': 512-16}, dataset=args.dataset)

draft_cache_budget = args.draft_cache_budget
recent_size = draft_cache_budget - 16
draft_cache = EvictStreamLLMCache(draft, start_size=16, recent_size=recent_size)
target_cache = SimpleCache(target, max_budget=prefill+gen_len+16)


####### Warm up for baseline ########
import math
import time

@torch.inference_mode()
def baseline(target, target_cache, max_len):
    target_cache.reset()
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in (range(iter_prefill)):
        # print(f"prefill {i}, {iter_prefill}, {input_ids[:, i*100:(i+1)*100]}")
        outputs = target(
            input_ids=input_ids[:, i*100:(i+1)*100].to(target.device),
            past_key_values=target_cache,
        )
    
    next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

    n = 0
    time1 = time.time()
    while n < max_len:
        logits = target(
            input_ids=next_token,
            past_key_values=target_cache,
        ).logits
        # print(next_token)
        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        n += 1
    time2 = time.time()
    # print(n, n / (time2 - time1))
    return n / (time2 - time1)

n_warmups = 1
input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Baseline Warmup"):
    baseline(target, target_cache, gen_len)

all_speed = []
for input_ids in tqdm(tokenized_prompts[:1], desc="Baseline Test"):
    input_ids = input_ids.to(target.device)[:,:prefill]
    speed = baseline(target, target_cache, gen_len)
    all_speed.append(speed)

baseline_latency = 1/(sum(all_speed) / len(all_speed))
print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} s", "red"))


all_acceptance_rate = []
all_latency = []
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

for input_ids in tqdm(tokenized_prompts):
    if prefill < 4096:
        if prefill == 1:
            input_ids = input_ids.to(draft.device)[:,2048:prefill+2048]
        else:
            input_ids = torch.cat([torch.LongTensor([[1]]).to(draft.device), input_ids.to(draft.device)[:,2048:prefill-1+2048]], dim=-1)
    else:
        input_ids = input_ids.to(draft.device)[:,:prefill]
    target_cache.reset()
    draft_cache.reset()
    acceptance_rate, latency = Evict_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=file_path, dataset=args.dataset, baseline=baseline_latency)
    all_acceptance_rate.append(acceptance_rate)
    all_latency.append(latency)

print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
print(colored(f"average latency: {sum(all_latency) / len(all_latency)}", "red"))
print(colored(f"[E2E Speedup]: {baseline_latency / (sum(all_latency) / len(all_latency))}", "red"))