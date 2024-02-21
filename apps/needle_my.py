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

import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--max_tokens', type=int, default=132000, help='maximum token length for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--max_length', type=int, default=131072, help='maximum token length of model')
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

if __name__ == "__main__":
    args = parse_args()
    
    model_name = 'LargeWorldModel/LWM-Text-1M'
    tokenizer = AutoTokenizer.from_pretrained('LargeWorldModel/LWM-Text-1M', use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

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