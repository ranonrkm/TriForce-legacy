# Written by Yukang Chen
# Core code based on https://github.com/CStanKonrad/long_llama
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import re
import token
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
import transformers

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from models.modeling_llama_ori import LlamaForCausalLM

from models.modeling_llama_flash import LlamaForCausalLM as LlamaForCausalLMFlash
from models.cache_utils import FlashStreamLLMCache
from utils.sampling import norm_logits, sample, max_fn
from termcolor import colored
import time

def good_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=1024, help='interval for evaluation')
    parser.add_argument('--max_tokens', type=int, default=102400, help='maximum token length for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--max_length', type=int, default=102400, help='maximum token length of model')
    return parser.parse_args()


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    #n_garbage_prefix = n_garbage // 2
    n_garbage_suffix = n_garbage - n_garbage_prefix
 
    task_description = "You are a helpful assistant. USER: There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    # pass_key = random.randint(1, 50000)
    pass_key = random.randint(1000000000000000000, 9000000000000000000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key?   Don't give information outside the document or repeat your findings. Keep your response short and direct. ASSISTANT: The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, device, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids[0]
    
    #answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    #pos = torch.where(tokenized_prompt == answer_ids[0][0])[0][0].item()
    
    if len(tokenized_prompt) > args.max_length:
        half = int(args.max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS

    # generation_output = model.generate(
    #     input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=True
    # )

    # print(torch.cat([input_ids, answer_ids[0]], dim=1))
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], do_sample=False, use_cache=True, num_beams=1
    )

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    # print(answer_ids, tokenizer.decode(answer_ids[0].cpu()))

    is_correct = (model_answer == answer_ids[0]).all().item()
    # print(model_answer, answer_ids[0])
    # print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    assert answer_ids[0].shape == model_answer.shape
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}, The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, len_token

def passkey_retrieval_test_streamllm(model, tokenizer, device, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids[0]
    
    #answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    #pos = torch.where(tokenized_prompt == answer_ids[0][0])[0][0].item()
    
    if len(tokenized_prompt) > args.max_length:
        half = int(args.max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half])+tokenizer.decode(tokenized_prompt[-half:])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS

    ############ StreamLLM cache ############
    start_size = len_token // 1024 + 26
    recent_size = int(len_token*0.1) - start_size
    past_key_values = FlashStreamLLMCache(model=model, max_budget=len_token+answer_ids.shape[-1]+10, skip_start_layers=1, start_size=start_size, recent_size=recent_size, gamma=4)

    gamma=1
    temperature = 0.6
    top_k = 1
    top_p = 0.9
    verbose = True

    with torch.no_grad():
        iter_prefill = math.ceil(input_ids.shape[1] / 100)
        for i in (range(iter_prefill)):
            outputs = model(
                input_ids=input_ids[:, i*100:(i+1)*100],
                past_key_values=past_key_values,
                use_cache=True,
                speculation=False,
            )

        past_key_values.print_status()

        max_len = answer_ids.shape[-1]
        resample_count = 0
        accepted_count = 0
        target_sample_count = 0
        draft_count = 0

        next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

        n = 0
        ans = [next_token.item()]

        time1 = time.time()
        while n < max_len:
            if next_token.shape == torch.Size([1]):
                next_token = next_token.unsqueeze(0)
            
            pred_token_idx = next_token
            
            # speculative decoding
            speculation_probs = []
            generated_ids = []

            past_key_values.update_cache()

            for _ in range(gamma):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    speculation=True,
                )
                probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
                pred_token_idx = sample(probs)
                
                speculation_probs.append(probs[0])
                generated_ids.append(pred_token_idx.item())

                draft_count += 1

            # verification
            verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(model.device)], dim=1)
            past_key_values.seq_len -= gamma

            outputs = model(
                input_ids=verify_tokens,
                past_key_values=past_key_values,
                use_cache=True,
                speculation=False,
            )

            count = 0
            verify_probs = []
        
            for i in range(gamma + 1):
                verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

            for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
                r = torch.rand(1, device = model.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                    count += 1
                    accepted_count += 1
                    n += 1
                    ans.append(i)
                    pred_token_idx = torch.tensor([[i]]).to(model.device)
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
                    ans.append(pred_token_idx.item())
                    if verbose:
                        good_stream(pred_token_idx, tokenizer, 'red')
                    break

                if tokenizer.eos_token_id == pred_token_idx:
                    break
            
            if count == len(generated_ids):
                target_sample_count += 1
                n += 1
                pred_token_idx = sample(verify_probs[-1])
                ans.append(pred_token_idx.item())
                if verbose:
                    good_stream(pred_token_idx, tokenizer, 'blue')

            next_token = pred_token_idx
            past_key_values.seq_len -= (gamma - count)
        time2 = time.time()
        accepted_rate = accepted_count / draft_count
        avg_tokens = accepted_count / draft_count * gamma


    is_correct = (torch.tensor(ans[:len(answer_ids[0])]) == answer_ids[0]).all().item()
    print(tokenizer.decode(ans[:len(answer_ids[0])]), tokenizer.decode(answer_ids[0]))
    # print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    # print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    print(max_len / (time2 - time1), "tokens/s, ", (time2 - time1) / max_len, "s/token", 'Sentence Length:', past_key_values.seq_len, f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
    with open(f"report/pw.csv", 'a') as f:
            f.write(f"{len_token},{accepted_rate},{(accepted_count)/ draft_count * gamma}\n")
    return is_correct, len_token

        # print(tokenizer.decode(ans[:len(answer_ids[0])]), tokenizer.decode(answer_ids[0]))
        # print(ans, answer_ids[0])

    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens, Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"generated tokens numbers {n}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
    
    else:
        print(max_len / (time2 - time1), "tokens/s, ", (time2 - time1) / max_len, "s/token", 'Sentence Length:', past_key_values.seq_len, f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
    

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

    is_correct = (model_answer == answer_ids[0]).all().item()
    # print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    # print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, len_token

if __name__ == "__main__":
    args = parse_args()
    
    model_name = 'LargeWorldModel/LWM-Text-1M'
    # model_name = "/home/hanshis/workspace/LongContextInfer/output/8192/step_5000"
    # model_name = "/home/hanshis/workspace/LongContextInfer/output/1024/step_30000"
    
    # model_name = "JackFram/llama-68m"
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m", use_fast=True, legacy=False)


    # model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda:0")

    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    
    # model = LlamaForCausalLMFlash.from_pretrained(model_name, device_map="auto")

    # model.resize_token_embeddings(model.config.vocab_size + 1)

    model.half().eval()

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        passed_tests = 0
        total_tokens = 0
        for j in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, model.device, n_garbage=n_garbage, seed=j)
            # is_correct, len_tokens = passkey_retrieval_test_streamllm(model, tokenizer, model.device, n_garbage=n_garbage, seed=j)
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens//args.num_tests
        accuracy = float(passed_tests)/args.num_tests
        print("accuracy on the token length %d is %f"%(avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)