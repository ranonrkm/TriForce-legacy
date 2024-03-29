import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

print(f"added {root_dir} to sys.path")

import torch
from transformers import AutoTokenizer
from termcolor import colored
import math
from models.modeling_llama_evict import LlamaForCausalLM
from models.cache_utils import SimpleCache, EvictStreamLLMCache, StreamLLMCache, EvictH2OCache
from termcolor import colored
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--method', type=str, default='streamingllm')
    parser.add_argument('--dataset', type=str, default='needle_retrieval_cached')
    parser.add_argument('--datalen', type=int, default=127*1024)
    return parser.parse_args()

args = parse_args()
budget = args.budget

######## model initialization ########
target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
target = target.eval()
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m", legacy=False, use_fast=True)

######## sampling parameters ########


top_k = 1
top_p = 1
temperature = 1

from data.dataset import get_dataset
tokenized_prompts, ans = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.datalen)

prefill = args.datalen
gen_len = 32
verbose = True
if args.method == 'streamingllm':
    target_cache = EvictStreamLLMCache(target, start_size=512, recent_size=budget-512)
elif args.method == 'h2o':
    target_cache = EvictH2OCache(target, start_size=budget//2, recent_size=budget-budget//2)
elif args.method == 'simple':
    target_cache = SimpleCache(target, budget)
all_acceptance_rate = []
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))
hit_list = []

with torch.inference_mode():
    for input_ids, ans_ in zip(tokenized_prompts, ans):
        target_cache.reset()
        ############ Iterative Pre-fill ############
        if args.method == 'streamingllm' or args.method == 'simple':
            T = 1024
        else:
            T = 128
        iter_prefill = math.ceil(input_ids.shape[1] / T)
        for i in range(iter_prefill):
            target_cache.evict(T)
            outputs = target(
                input_ids=input_ids[:, i*T:(i+1)*T].to(target.device),
                past_key_values=target_cache,
            )
        target_cache.print_status()
        ############ Iterative Generation ############
        next_token = torch.argmax(outputs.logits[:,-1,:], dim=-1)

        gen_tokens = []
        gen_tokens.append(next_token.item())

        for _ in range(gen_len):
            target_cache.evict(1)
            outputs = target(
                input_ids=next_token.unsqueeze(0).to(target.device),
                past_key_values=target_cache,
            )
            next_token = torch.argmax(outputs.logits[:,-1,:], dim=-1)
            # print(tokenizer.decode(next_token[0].tolist()))
            gen_tokens.append(next_token.item())
        
        print('model ans:', tokenizer.decode(gen_tokens)[:32])
        print('ans:', ans_)
        is_correct = (tokenizer.decode(gen_tokens)[:32] == ans_[:32])
        print(is_correct)
        hit_list.append(is_correct)

    print(f"Hit rate: {sum(hit_list)/len(hit_list)} with budget {budget} for method {args.method}" )
    with open(f"needle_test.csv", "a") as f:
        f.write(f"{args.method},{budget},{sum(hit_list)/len(hit_list)}\n")