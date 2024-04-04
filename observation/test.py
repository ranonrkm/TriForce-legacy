import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from models.modeling_observation import LlamaForCausalLM
from models.cache_utils import FlashSimpleCache
import math
from utils.sampling import norm_logits, sample
from utils.misc import spec_stream
import socket
host = socket.gethostname()
import argparse
import gc
import csv
def sparsity(scores, tops = [1,8,16,32,64,128,256, 512, 1024, 2048,4096, 8192, 16384], gen_len=256):
    ret = {}
    for layer in tqdm(range(len(scores))):
        for top in tops:
            token_list = []
            for token in range(gen_len):
                topk_values, _ = torch.topk(scores[layer][token], k=top, dim=-1)
                token_list.append(topk_values.sum(dim=-1) / scores[layer][token].sum(dim=-1))
            
            tmp = sum(token_list) / len(token_list)
            ret[f"layer-{layer}-top-{top}"] = tmp.cpu().numpy()
    
    return ret

def locality(scores, prefill, gen_len):
    long_info = {}
    for layer in range(32):
        long_info[layer] = []
        for token in range(gen_len):
            _, topk_idx = torch.topk(scores[layer][0][:,:prefill], k=4096, dim=-1)
            selected_scores = torch.gather(scores[layer][token][:,:prefill], 1, topk_idx)
            restored_long = (selected_scores.sum(dim=-1) / scores[layer][token][:,:prefill].sum(dim=-1)).mean()
            long_info[layer].append(restored_long.item())
    return long_info

######## model initialization ########
target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
target = target.eval()
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='benchmark', tokenizer=tokenizer, datalen=122880)

######## sampling parameters ########

top_k = -1
top_p = 0.9
temperature = 0.7

prefill = 1024
gen_len = 1024

####### cache init #######
print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

######## Warm up for baseline ########
with torch.inference_mode():
    for input_ids in tqdm(tokenized_prompts, desc="Getting..."):
        cache = FlashSimpleCache(target, prefill+gen_len+16)
        cache.reset()
        cache.print_status()
        input_ids = input_ids.to(target.device)[:,:prefill]

        iter_prefill = math.ceil(input_ids.shape[1] / 100)
        for i in range(iter_prefill):
            logits = target(
                input_ids=input_ids[:, i*100:(i+1)*100],
                kv_cache=cache,
                graph_cache=None,
            ).logits
        cache.print_status()

        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

        n = 0
        while n < gen_len:
            logits = target(input_ids=next_token, kv_cache=cache).logits
            # print(next_token)
            next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
            n += 1
            spec_stream(next_token[0], tokenizer, 'cyan')

        cache.print_status()
        print(colored(f"{'='*80}", "red"))

        sparsity_dict = sparsity(scores=cache.scores, gen_len=gen_len)
        long_info = locality(cache.scores, prefill, gen_len)
        
        del cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        with open('sparsity.csv', 'w') as f:
            for key in sparsity_dict.keys():
                f.write("%s,%s\n"%(key,sparsity_dict[key]))

        with open('locality.csv', 'a') as f:
            for key in long_info.keys():
                f.write("%s,%s\n"%(key,long_info[key]))