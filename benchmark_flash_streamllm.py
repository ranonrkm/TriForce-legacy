import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
import torch
import math
from tqdm import tqdm
import time

from models.cache_utils import FlashSimpleCache, FlashStreamLLMCache
from models.modeling_llama_flash import LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='auto')
model = model.eval()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--datalen', type=int, default=128000, help='length of data')
    parser.add_argument('--budget', type=float, default=0.1, help='budget of cache')
    parser.add_argument('--ssl', type=int, default=1, help='skip_layers')
    parser.add_argument('--T', type=int, default=2000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()


data_len = args.datalen
ssl = args.ssl
budget = args.budget
past_key_values = FlashStreamLLMCache(model, data_len+200, start_size=64, recent_size=int((data_len+200)*budget)-64, skip_start_layers=ssl, gamma=130)

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')
input_ids = tokenized_prompts[0].to(model.device)[:,:data_len]
past_key_values.reset()

T=args.T
# from 1 t 128
# LEN = [1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128]

LEN = [1]

l = 128
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
    for _ in range(10):
        outputs = model(
            input_ids=sentence,
            past_key_values=past_key_values,
            use_cache=True,
            speculation=True,
        )
        past_key_values.seq_len -= l
        past_key_values.spec_time = 0
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time / T, l, data_len, T, "warm up done")

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

    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        past_key_values.update_cache()
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)
    update_time = total_time / T

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
                speculation=True,
            )
            past_key_values.seq_len -= l
            past_key_values.spec_time = 0
            # assert past_key_values.seq_len == data_len
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
    
        print(total_time / T, update_time, l, data_len, T)
    
        if budget != 0.1:
            with open(f"report/EXP_benchmark_flash_streamllm_{budget}.csv", 'a') as f:
                f.write(f"{data_len},{l},{total_time / T},{update_time},{ssl},{T}\n")

        else:
            # write to file
            with open(f"report/EXP_benchmark_flash_streamllm.csv", 'a') as f:
                f.write(f"{data_len},{l},{total_time / T},{update_time},{ssl},{T}\n")

