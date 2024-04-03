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
from torch.nn.functional import softmax
from utils.tree_infer import GraphInferenceEngine, get_sampling_logits, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement, get_residual
from utils.SpecTree_TP import SpecTree

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)
model_name_or_path = "NousResearch/Yarn-Llama-2-7b-128k"

def create_sampling_callable(num_samples, temperature=0.6):
    def sampling_without_replacement(sampling_logits: torch.Tensor, static_rand):
        if torch.distributed.get_rank() == 0:
            sampling_q = softmax(sampling_logits / temperature, dim=-1)
            position = (static_rand.log()/sampling_q).topk(k=num_samples).indices.flatten()
        else:
            position = torch.full((num_samples * sampling_logits.shape[0],), -1, dtype=torch.long, device=sampling_logits.device)
        torch.distributed.broadcast(position, src=0)
        return position
    
    return sampling_without_replacement

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--dataset', type=str, default='benchmark', help='dataset')
    parser.add_argument('--budget', type=int,  default=5120)
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--tree_size', type=str, default='1024')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

prefill = args.prefill
gen_len = args.gen_len
temperature = 0.6
top_p = 0.9
retrieval_budget = args.budget

####### tree #######
residual_graph = get_residual
path = f'tree/{args.tree_size}.pt'

grow_map = torch.load(path)
tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, legacy=False)
llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size, prefill=prefill, gen_len=gen_len, temperature=temperature, top_p=top_p, flash_attn=True, retrieval_budget=retrieval_budget, kv_offload=True, on_chip_layers=8, tree_size=tree_size)
for rank in range(world_size):
    if local_rank == rank:
        print(f"Rank {rank+1}/{world_size} (Device {device}) is initializing parameters")
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=32768)
input_ids = tokenized_prompts[0][:,:prefill].to(device)

# baseline_latency, gen_tokens = Baseline_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
# baseline_latency = baseline_latency/1000
# if local_rank == 0:
#     llm.kv_cache.print_status()
#     print(tokenizer.batch_decode(gen_tokens))
#     print(colored(f"[Baseline-Autoregressive] average latency: {baseline_latency} s", "red"))
# dist.barrier()
baseline_latency = 1
sampling_callables = {}
for i in range(draft_step - 1):
    num_samples = max(branch_lists[i])
    sampling_callables[i] = create_sampling_callable(num_samples=num_samples, temperature=temperature)

sample_gather_indices = {}
for i in range(draft_step - 1):
    ith_gather_list = []
    max_num_samples = max(branch_lists[i])
    for j, branch in enumerate(branch_lists[i]):
        branch_index = torch.arange(branch, device=llm.device, dtype=torch.long)
        branch_index = branch_index + j * max_num_samples
        ith_gather_list.append(branch_index)
    ith_gather_list = torch.cat(ith_gather_list)
    sample_gather_indices[i] = ith_gather_list

if local_rank == 0:
    print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

######## Our method ########
all_speed_up = []
all_acc_list = []

dtype = torch.float16
max_length = prefill+gen_len
spectree = SpecTree(engine=llm, temperature=args.temp, top_p=top_p,
                    max_length=prefill+gen_len, grow_map=grow_map,
                    residual_graph=residual_graph,
                    sampling_callables=sampling_callables,
                    sample_gather_indices=sample_gather_indices,
                    tokenizer=tokenizer)

for input_ids in tokenized_prompts:
    input_ids = input_ids[0,:args.prefill].to(llm.device)

    with torch.inference_mode():
        n=0
        pos = 0
        generated_ids = []
        
        next_token = spectree.prefill(prefix=input_ids)
        acc_count_list = []
        generated_ids.extend(next_token[0].tolist())

        time1 = time.time()
        while n < gen_len:
            spectree.construct_grow_map(next_token=next_token)
            next_token, acc_count, print_tokens = spectree.verify()
            
            generated_ids.extend(print_tokens[1:].tolist())
            
            generated_text = (
                tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
            )

            if local_rank == 0:
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token is None:
                break
            next_token = next_token.unsqueeze(0)
            n += acc_count
            acc_count_list.append(acc_count)
        if local_rank == 0:
            print(" ".join(generated_text[pos:]), flush=True)
        if n < 64:
            continue
        torch.cuda.synchronize()
        time2 = time.time()
        method_latency = (time2 - time1)/n
        dist.barrier()
        if local_rank == 0:
            print(f"[Avg Accepted Tokens]: {np.array(acc_count_list).mean()}")
            print(colored(f"[Ours-Chain_Retrieval] average latency: {method_latency} s", "red"))
        # print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))

    all_acc_list.append(np.array(acc_count_list).mean())
    all_speed_up.append(baseline_latency / method_latency)
# print(f"[Overall Speedup]: {np.array(all_speed_up).mean()}")
print(f"[Overall Avg Accepted Tokens]: {np.array(all_acc_list).mean()}")

# destory the distributed process
dist.destroy_process_group()

# acceptance_rate, avg_tokens, retrieval_latency, gen_tokens = Retrieval_Spec_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
# if local_rank == 0:
#     # print(tokenizer.batch_decode(gen_tokens))
#     llm.kv_cache.print_status()
#     print(colored(f"[Retrieval-Speculative-Decoding] average latency: {retrieval_latency} ms", "red"))
#     all_speed_up.append(baseline_latency / retrieval_latency)
#     all_acc_list.append(acceptance_rate)
# dist.barrier()

# print(f"[Overall Speedup]: {np.array(all_speed_up).mean()}")
# print(f"[Overall Avg Accepted Tokens]: {np.array(all_acc_list).mean()}")