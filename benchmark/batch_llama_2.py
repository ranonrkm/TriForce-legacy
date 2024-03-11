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
from models.batch_cache import BatchSimpleCache
from models.modeling_batch_llama import LlamaForCausalLM
from utils.batch_infer import GraphInferenceEngine
from utils.sampling import sample, norm_logits, max_fn

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')
model = model.eval()

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
cache = BatchSimpleCache(model, data_len+256+16, bsz=bsz)
cache.reset()


T=args.T
graph_engine = GraphInferenceEngine(model, cache, None, model, None)
# set kv length
seq_len = torch.full((bsz,), data_len, dtype=torch.int32).cuda()
# seq_len = torch.randint(low=data_len-256, high=data_len, size=(bsz,), dtype=torch.int32).cuda()
graph_engine.engine.kv_cache.set_len(seq_len)
cache.print_status()

# warm up
with torch.inference_mode():
    sentence = torch.randint(low=3, high=30000, size=(bsz, 1)).to(model.device)
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        graph_engine.inference(sentence)
        graph_engine.engine.kv_cache.seq_len -= 1
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)


LEN = [1,2,4,5,6,7,8,9,10,11,12,12,13,14,15,16,32,64,96,128]

with torch.inference_mode():
    for l in LEN:
        graph_engine.engine.kv_cache.set_len(seq_len)
        
        sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(model.device)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
            graph_engine.inference(sentence)
            graph_engine.engine.kv_cache.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        foward_time = (t2 - t1) / T * 1000

        logits = torch.randn(bsz, l, 32000).half().cuda()
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
            next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        torch.cuda.synchronize()
        t2 = time.time()
        sampling_time = (t2 - t1) / T * 1000

        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(256):
            logits = graph_engine.inference(next_token)
            next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        torch.cuda.synchronize()
        t2 = time.time()
        all_time = (t2 - t1) / 256 * 1000

        print(f"bsz={bsz}, prefill={data_len}, verify_len={l}, foward_time={foward_time}, sampling_time={sampling_time}, decoding_time={all_time}")

