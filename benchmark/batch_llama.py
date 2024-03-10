import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import socket

from transformers import AutoTokenizer, GPTQConfig
import torch
import math
from tqdm import tqdm
import time

from data.dataset import get_dataset
from models.cache_utils import BatchSimpleCache
from models.modelding_llama_batch import LlamaForCausalLM
from utils.batch_infer import GraphInferenceEngine

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')
model = model.eval()

host = socket.gethostname()
if 'cr-a100-80-0004' in host:
    file_path = "/var/cr06_data/beidic/LongContextInfer/benchmark/report/A100_llama_7B_128K_batch.csv"
else:
    file_path = "/fsx-storygen/beidic/hanshi/LongContextInfer/benchmark/report/A100_llama_7B_128K_batch.csv"

try:
    with open(file_path, 'r') as f:
        contents = f.read()
except FileNotFoundError:
    contents = ""

if not contents:
    with open(file_path, 'a') as f:
        f.write("prefill,len,latency,repeat_time\n")

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--bsz', type=int, default=2, help='bsz')
    parser.add_argument('--T', type=int, default=1000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

bsz = args.bsz
data_len = int(1024*120 / bsz)
cache = BatchSimpleCache(model, data_len+128, bsz=bsz)
cache.reset()
cache.print_status()

# warm up
T=args.T
l=1

graph_engine = GraphInferenceEngine(model, cache, None, model, None)
graph_engine.engine.kv_cache.seq_len = data_len

with torch.inference_mode():
    sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(model.device)
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        graph_engine.inference(sentence)
        graph_engine.engine.kv_cache.seq_len -= l
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time *10, l, data_len, 100, "warm up done")


LEN = [1,2,4,8,16,32,64,96,128]

# LEN = [1]
with torch.no_grad():
    for l in LEN:
        sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(model.device)
        total_time = 0.0
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(1000):
            graph_engine.inference(sentence)
            graph_engine.engine.kv_cache.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)

        print(round(total_time,4), l, data_len, 1000, "test")
        
        # write to file
        # with open(file_path, 'a') as f:
        #     f.write(f"{data_len},{l},{total_time / T},{T}\n")

