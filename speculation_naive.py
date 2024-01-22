
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8,9"
from termcolor import colored

from transformers import AutoTokenizer
from models.modeling_llama_cache import LlamaForCausalLM

import torch
from utils.sampling import norm_logits, sample, max_fn
import time
from models.cache_utils import SimpleCache
from tqdm import tqdm
import math

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--approx', type=str, default='llama-1.1B-32K', help='approx model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--greedy', action='store_true', help='greedy')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

print(f"Using {args.approx} to speculate {args.target}", flush=True)

######## model initialization ########

if args.approx == 'llama-1.1B-32K':
    model_small = LlamaForCausalLM.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

model_small = model_small.eval()

if args.target == 'llama-7B-128K':
    model_target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
else:
    raise NotImplementedError

model_target = model_target.eval()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")

######## sampling parameters ########

if args.greedy:
    top_k = 1
    top_p = 0.9
    temperature = 1
else:
    top_k = -1
    top_p = 0.9
    temperature = 0.6

def good_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

from data.dataset import get_dataset

tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')

verbose = args.verbose

data_len=32000

print(f"Data Length: {data_len}", flush=True)

for input_ids in tokenized_prompts:
    input_ids = input_ids.to(model_small.device)[:,:data_len]

    past_key_values_small = SimpleCache(model_small, max_budget=data_len+300)
    past_key_values_target = SimpleCache(model_target, max_budget=data_len+300)

    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        with torch.no_grad():
            outputs = model_small(
                input_ids=input_ids[:, i*100:(i+1)*100],
                past_key_values=past_key_values_small,
                use_cache=True,
                speculation=False,
            )
            outputs = model_target(
                input_ids=input_ids[:, i*100:(i+1)*100],
                past_key_values=past_key_values_target,
                use_cache=True,
                speculation=False,
            )
    
    gamma = 4

    for i in range(1):
        max_len = 200
        resample_count = 0
        accepted_count = 0
        target_sample_count = 0
        draft_count = 0

        next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

        if verbose:
            good_stream(next_token[0], tokenizer, 'cyan')

        n = 0
        time1 = time.time()
        while n < max_len:
            if next_token.shape == torch.Size([1]):
                next_token = next_token.unsqueeze(0)
            
            pred_token_idx = next_token
            
            # speculative decoding
            speculation_probs = []
            generated_ids = []

            for _ in range(gamma):
                with torch.no_grad():
                    outputs = model_small(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values_small,
                        use_cache=True,
                        speculation=False,
                    )

                probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
                pred_token_idx = sample(probs)
                speculation_probs.append(probs[0])
                
                generated_ids.append(pred_token_idx.item())

                draft_count += 1

            # verification
            verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(model_small.device)], dim=1)

            with torch.no_grad():
                outputs = model_target(
                    input_ids=verify_tokens,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    speculation=False,
                )

            count = 0
            verify_probs = []
        
            for i in range(gamma + 1):
                assert outputs.logits.shape[1] == gamma + 1
                verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

            for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
                r = torch.rand(1, device = model_small.device)

                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                    count += 1
                    accepted_count += 1
                    n += 1
                    pred_token_idx = torch.tensor([[i]]).to(model_small.device)
                    if verbose:
                        good_stream(i, tokenizer, 'green')

                    # if eos
                    if tokenizer.eos_token_id == i:
                        draft_count -= gamma - count
                        break

                else:
                    resample_count += 1
                    n += 1
                    pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                    if verbose:
                        good_stream(pred_token_idx, tokenizer, 'red')
                    break

            # if eos
            if tokenizer.eos_token_id == pred_token_idx:
                break

            if count == len(generated_ids):
                target_sample_count += 1
                n += 1
                pred_token_idx = sample(verify_probs[-1])
                if verbose:
                    good_stream(pred_token_idx, tokenizer, 'blue')

            next_token = pred_token_idx
            
            if gamma - count > 0:
                past_key_values_small.seq_len -= (gamma - count) - 1
            else:
                with torch.no_grad():
                    outputs = model_small(
                        input_ids=torch.tensor([[generated_ids[-1]]]).to(model_small.device),
                        past_key_values=past_key_values_small,
                        use_cache=True,
                        speculation=False,
                    )

            past_key_values_target.seq_len -= (gamma - count)
            assert past_key_values_target.seq_len == past_key_values_small.seq_len, f"{past_key_values_target.seq_len} != {past_key_values_small.seq_len}"

            # print(past_key_values_target.seq_len)


        time2 = time.time()
        accepted_rate = accepted_count / draft_count
        avg_tokens = accepted_count / draft_count * gamma
        if verbose:
            print(f"Use {time2 - time1} sec to generate {n} tokens, Tokens/s: {n / (time2 - time1)}", flush=True)
            print(f"generated tokens numbers {n}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
            print(f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
        else:
            print(max_len / (time2 - time1), "tokens/s, ", (time2 - time1) / max_len, "s/token", 'Sentence Length:', past_key_values_target.seq_len, f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")

