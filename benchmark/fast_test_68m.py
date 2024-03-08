import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from transformers import AutoTokenizer
from models.modeling_llama_torch import LlamaForCausalLM
from models.cache_utils import SimpleCache, EvictStreamLLMCache
import torch
import numpy as np 
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
import argparse
from data.dataset import get_dataset
import math

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=1.0, help='top_p')
parser.add_argument('--DP', type=float, default=1.1, help='draft_top_p')
parser.add_argument('--W', type=int, default=16, help='max width')
parser.add_argument('--dst', type=str, default="68m.pt", help='destination for accepetance rate vector')
args = parser.parse_args()
print(args)
def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    
    return residual

def evaluate(target, draft, cache, draft_cache, prompts, prefill, k:int, T=0.6, top_p=0.9, draft_top_p=0.99):
    acceptance_rate = torch.zeros(k)
    num_samples = 0
    draft_model_prob = []
    token_accept_rate = []
    sampled_token_sets = []
    real_budget = 0
    with torch.no_grad():
        for input_ids in prompts:
            cache.reset()
            draft_cache.reset()
            
            input_ids = input_ids.to(target.device)
            prefill_ids = input_ids[:, :prefill]
            input_ids = input_ids[:, prefill:prefill+128]

            iter_prefill = math.ceil(prefill_ids.shape[1] / 100)
            for i in tqdm(range(iter_prefill)):
                outputs = target(
                    input_ids=prefill_ids[:, i*100:(i+1)*100].to(target.device),
                    past_key_values=cache,
                )

                draft_cache.evict(100)
                outputs_draft = draft(
                    input_ids=prefill_ids[:, i*100:(i+1)*100].to(draft.device),
                    past_key_values=draft_cache,
                )

            target_logits : torch.Tensor = target(input_ids=input_ids, past_key_values=cache).logits

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                target_logits[indices_to_remove] = float('-inf')

            draft_cache.evict(128)
            draft_logits : torch.Tensor = draft(input_ids=input_ids, past_key_values=draft_cache).logits
            target_prob = softmax(target_logits / T, dim=-1).squeeze(0)
            q = softmax(draft_logits / T, dim=-1).squeeze(0)
            
            for i in range(20):
                token_acceptance_rate = torch.zeros(k)
                draft_tokens = []
                num_samples = num_samples + 1
                token_target_prob = target_prob[i]
                # token_draft_prob = q[i]
                #draft_model_prob.append(q[i].cpu())
                token_draft_logits = draft_logits[0][i]

                if draft_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_draft_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                    filter = cumulative_probs > draft_top_p
                    filter[..., 1:] = filter[..., :-1].clone()
                    filter[..., 0] = 0
                    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                    token_draft_logits[indices_to_remove] = float('-inf')

                token_draft_prob = softmax(token_draft_logits / T, dim=-1).squeeze(0)
                sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                draft_tokens.append(sampled_token.item())
                real_budget = real_budget + 1
                token_acceptance_rate[0] = min(1.0, (token_target_prob[sampled_token]/ token_draft_prob[sampled_token]))

                token_target_prob = get_residual(token_target_prob, token_draft_prob)
                
                
                for j in range(k-1):
                    token_draft_logits[sampled_token] = - torch.inf
                    token_draft_prob = softmax(token_draft_logits / (T), dim=-1).squeeze(0)
                    if torch.isnan(token_draft_prob).long().sum() >= 1:
                        break
                    token_draft_prob = token_draft_prob / token_draft_prob.sum(-1)
                    sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                    draft_tokens.append(sampled_token.item())
                    real_budget = real_budget + 1
                    branch_token_acceptance_rate = min(1, token_target_prob[sampled_token]/ token_draft_prob[sampled_token])
                    token_acceptance_rate[j+1] = (1 - token_acceptance_rate.sum()) * branch_token_acceptance_rate
                    
                    token_target_prob = get_residual(token_target_prob, token_draft_prob)
                acceptance_rate = acceptance_rate + token_acceptance_rate
                token_accept_rate.append(token_acceptance_rate.cpu())
                sampled_token_sets.append(draft_tokens)
                draft_model_prob.append(q[i][draft_tokens].cpu()) 
            print(acceptance_rate, num_samples)
    return acceptance_rate / num_samples


prefill=122880
gen_len=128
target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
target = target.eval()
draft = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="auto")
draft = draft.eval()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)
tokenized_prompts = get_dataset(dataset_name='128k', tokenizer=tokenizer, datalen=prefill)

acceptance_rate_list = [0]
branch_acceptance_rate_list = [0]

draft_cache_budget = 256
recent_size = draft_cache_budget - 16
draft_cache = EvictStreamLLMCache(draft, start_size=16, recent_size=recent_size)
cache = SimpleCache(target, max_budget=prefill+gen_len+16)

acceptance_rate = evaluate(target, draft, cache, draft_cache, tokenized_prompts, prefill, k=args.W, T=args.T, top_p=args.P, draft_top_p=args.DP)
x = torch.zeros(len(acceptance_rate) + 1)
x[1:] = acceptance_rate
torch.save(x, args.dst)
print(x)
