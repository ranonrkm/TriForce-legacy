import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from transformers import AutoTokenizer
import torch
import math
from tqdm import tqdm
import time

from models.cache_utils import FlashSimpleCache
from models.modeling_llama_flash import LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')
model = model.eval()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--datalen', type=int, default=128000, help='length of data')
    parser.add_argument('--T', type=int, default=2000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()


data_len = args.datalen
past_key_values = FlashSimpleCache(model, data_len+1200)
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')
input_ids = tokenized_prompts[0].to(model.device)[:,:data_len]
past_key_values.reset()


# warm up

T=args.T
l=1024

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

    print(total_time / T, l, data_len, T, "warm up done")

LEN = [1,2,4,8,16,32,64,128,256,512,1024]


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
    
        # write to file
        with open("report/benchmark_flash.csv", 'a') as f:
            f.write(f"{data_len},{l},{total_time / T},{T}\n")

