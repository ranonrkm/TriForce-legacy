import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer
import torch
import math
from tqdm import tqdm
import time

from models.cache_utils import FlashSimpleCache
from models.modeling_llama_flash import LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='auto')
model = model.eval()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--T', type=int, default=2000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()


data_len = 0
past_key_values = FlashSimpleCache(model, data_len+120)
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')
input_ids = tokenized_prompts[0].to(model.device)[:,:data_len]
past_key_values.reset()

T=args.T
LEN = [1]

with torch.no_grad():
    sentence = torch.randint(low=3, high=30000, size=(1, 1)).to(model.device)
    for i in tqdm(range(10)):
        outputs = model(
            input_ids=sentence,
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values.reset()
    past_key_values.print_status()

    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        outputs = model(
            input_ids=sentence,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values.seq_len -= 1
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time / T)

