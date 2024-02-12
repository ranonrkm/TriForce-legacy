import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored

from models.modeling_llama_torch import LlamaForCausalLM
from models.cache_utils import SimpleCache
from utils.decoding import Vanilla_Spec_cache
from utils.misc import print_config

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68m-256', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')

    parser.add_argument('--prefill', type=int, default=128, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=128, help='generation length')
    parser.add_argument('--gamma', type=int, default=1, help='gamma')
    parser.add_argument('--log_csv', action='store_true', help='log_csv')

    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

######## model initialization ########

if args.draft == 'llama-68m-256':
    draft = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="auto")
    # draft = LlamaForCausalLM.from_pretrained("/home/hanshis/workspace/legacy/output/256/step_30000", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

if args.target == 'llama-7B-128K':
    target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

draft = draft.eval()
target = target.eval()

tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")

######## sampling parameters ########

if args.greedy:
    top_k = -1
    top_p = 1
    temperature = 1
else:
    top_k = -1
    top_p = 0.9
    temperature = 0.6

from data.dataset import get_dataset

# tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen='4k')
tokenized_prompts = get_dataset(dataset_name='c4', tokenizer=tokenizer, datalen='4k')

prefill = args.prefill
gen_len = args.gen_len
gamma = args.gamma
verbose = args.verbose

if args.log_csv:
    import socket
    host = socket.gethostname()
    if 'lovelace' in host:
        file_path = "/home/hanshis/workspace/LongContextInfer/test/report/L40_vanilla.csv"
    else:
        file_path = "/fsx-storygen/beidic/hanshi/LongContextInfer/test/report/A100_vanilla.csv"
else:
    file_path = None

print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=file_path)

draft_cache = SimpleCache(draft, max_budget=prefill+gen_len+16)
target_cache = SimpleCache(target, max_budget=prefill+gen_len+16)

all_acceptance_rate = []

for input_ids in tokenized_prompts:
    input_ids = input_ids.to(draft.device)[:,:prefill]

    acceptance_rate = Vanilla_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=file_path)
    all_acceptance_rate.append(acceptance_rate)

print(colored(f"average acceptance rate: {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))