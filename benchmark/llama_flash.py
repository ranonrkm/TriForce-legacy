import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import socket

from transformers import AutoTokenizer
import torch
import math
from tqdm import tqdm
import time

from data.dataset import get_dataset
from models.cache_utils import FlashSimpleCache
from models.modeling_llama_flash import LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='auto')
model = model.eval()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--datalen', type=int, default=1024, help='length of data')
    parser.add_argument('--T', type=int, default=2000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()


data_len = args.datalen
past_key_values = FlashSimpleCache(model, data_len+600)
tokenized_prompt = get_dataset(dataset_name='benchmark', tokenizer=tokenizer)[0]
input_ids = tokenized_prompt.to(model.device)[:,:data_len]
past_key_values.reset()


# warm up

T=args.T
l=512

host = socket.gethostname()

if 'lovelace' in host:
    file_path = "/home/hanshis/workspace/LongContextInfer/benchmark/report/L40_llama_7B_128K_flash.csv"
else:
    file_path = "/fsx-storygen/beidic/hanshi/LongContextInfer/benchmark/report/A100_llama_7B_128K_flash.csv"

try:
    with open(file_path, 'r') as f:
        contents = f.read()
except FileNotFoundError:
    contents = ""

if not contents:
    with open(file_path, 'a') as f:
        f.write("prefill,len,latency,repeat_time\n")

with torch.no_grad():
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        outputs = model(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values.print_status()

    sentence = torch.randint(low=3, high=30000, size=(1, l)).to(model.device)
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        outputs = model(
            input_ids=sentence,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values.seq_len -= l
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time / 100, l, data_len, 100, "warm up done")


LEN = [1,2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512]

past_key_values.reset()
with torch.no_grad():
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        outputs = model(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values.print_status()

    for l in LEN:
        sentence = torch.randint(low=3, high=30000, size=(1, l)).to(model.device)
        total_time = 0.0
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
            outputs = model(
                input_ids=sentence,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
    
        print(total_time / T, l, data_len, T)
        past_key_values.print_status()
    
        # write to file
        with open(file_path, 'a') as f:
            f.write(f"{data_len},{l},{total_time / T},{T}\n")

