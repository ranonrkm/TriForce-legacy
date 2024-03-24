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
from models.batch_cache import BatchSimpleCache, BatchRetrievalCache
from models.llama_mqa import LlamaForCausalLM
from utils.batch_infer import GraphInferenceEngine
from utils.sampling import sample, norm_logits, max_fn

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-200K")
model = LlamaForCausalLM.from_pretrained("01-ai/Yi-6B-200K", torch_dtype=torch.float16, device_map='cuda:0')
model = model.eval()

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--bsz', type=int, default=2, help='bsz')
    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--attn_method', type=str, default='flash', help='attn_method')
    parser.add_argument('--T', type=int, default=1000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

T=args.T
gamma=6
bsz = args.bsz
data_len = args.prefill//8*8
attn_method = args.attn_method

cache = BatchSimpleCache(model, data_len+256+16, bsz=bsz)
graph_cache = BatchRetrievalCache(model, 1024, bsz=bsz, prefill=data_len, chunk_size=8, gamma=gamma)
cache.reset()

graph_engine = GraphInferenceEngine(model, cache, graph_cache, None, None, bsz)
graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=0.6, top_p=0.9)
# set kv length
seq_len = torch.full((bsz,), data_len, dtype=torch.int32).cuda()
# seq_len = torch.randint(low=data_len-256, high=data_len, size=(bsz,), dtype=torch.int32).cuda()
graph_engine.engine.kv_cache.set_len(seq_len)
graph_engine.engine.graph_cache.normal_()
cache.print_status()

# warm up
with torch.inference_mode():
    sentence = torch.randint(low=3, high=30000, size=(bsz, 1)).to(model.device)
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        graph_engine.engine.model(input_ids=sentence, kv_cache=graph_engine.engine.kv_cache, graph_cache=None, attn_method=attn_method).logits
        graph_engine.engine.kv_cache.seq_len -= 1
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)


LEN = [1,2,4,5,6,7,8,9,10,11,12,12,13,14,15,16,32,64,96,128]

with torch.inference_mode():
    for l in LEN:
        graph_engine.engine.kv_cache.set_len(seq_len)
        
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
            sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(model.device)
            graph_engine.engine.model(input_ids=sentence, kv_cache=graph_engine.engine.kv_cache, graph_cache=None, attn_method=attn_method).logits
            graph_engine.engine.kv_cache.seq_len -= l
        torch.cuda.synchronize()
        t2 = time.time()
        foward_time = (t2 - t1) / T * 1000

        torch.cuda.synchronize()
        t1 = time.time()
        position_ids = graph_engine.engine.kv_cache.seq_len[:, None] + gamma -1
        input_ids = torch.randint(low=3, high=30000, size=(bsz, 1)).to(model.device)
        for _ in range(T):
            sentence = torch.randint(low=3, high=30000, size=(bsz, l)).to(model.device)
            graph_engine.graph_retrieval_inference(input_ids=input_ids, gamma_offset=gamma -1, position_ids=position_ids)
        torch.cuda.synchronize()
        t2 = time.time()
        draft_time = (t2 - t1) / T * 1000

        # logits = torch.randn(bsz, l, 32000).half().cuda()
        # torch.cuda.synchronize()
        # t1 = time.time()
        # for _ in range(T):
        #     next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        # torch.cuda.synchronize()
        # t2 = time.time()
        # sampling_time = (t2 - t1) / T * 1000


        # graph_engine.engine.kv_cache.set_len(seq_len)
        # torch.cuda.synchronize()
        # t1 = time.time()
        # for _ in range(256):
        #     logits = graph_engine.engine.model(input_ids=sentence, kv_cache=graph_engine.engine.kv_cache, graph_cache=None).logits
        #     next_token = sample(norm_logits(logits[:,-1,:], temperature=0.6 ,top_k=-1, top_p=0.9))
        #     graph_engine.engine.kv_cache.seq_len -= l
        # torch.cuda.synchronize()
        # t2 = time.time()
        # all_time = (t2 - t1) / 256 * 1000

        print(f"bsz={bsz}, prefill={data_len}, verify_len={l}, foward_time={foward_time}, draft_time={draft_time}", flush=True)
        # print(f"bsz={bsz}, prefill={data_len}, verify_len={l}, foward_time={foward_time}, draft_time={draft_time}, sampling_time={sampling_time}, decoding_time={all_time}", flush=True)

