import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
import torch
import time
import argparse
import math
from tqdm import tqdm
import socket
from time import sleep
from torch.profiler import profile, record_function, ProfilerActivity
from models.modeling_llama_graph import LlamaForCausalLM, LlamaConfig
from models.cache_utils import SimpleCache, FlashSimpleCache, GraphFlashSimpleCache, GraphSimpleCache
from utils.graph_infer import GraphInferenceEngine

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=2000, help='time')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--flash', action='store_true', help='flash')
parser.add_argument('--model_name', type=str, default="NousResearch/Yarn-Llama-2-7b-128k", help='model name')
args = parser.parse_args()

PREFIX_LEN = args.P
T = args.T
WARM_UP = 10

model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')

DEC_LEN_LIST = [1]

MAX_LEN = PREFIX_LEN + 1

cache = FlashSimpleCache(model, 1)
graph_cache = GraphSimpleCache(model, MAX_LEN)

for DEC_LEN in DEC_LEN_LIST:
    cache.reset()
    graph_cache.reset()
    prefix = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=model.device)
    assert prefix.shape[-1] == PREFIX_LEN
    
    graph_engine = GraphInferenceEngine(model, cache, graph_cache)
    graph_engine.initialize_cuda_graph(1)

    input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=model.device)
    storage_ids = torch.arange(DEC_LEN, device=model.device) + PREFIX_LEN
    position_ids = storage_ids.clone().unsqueeze(0)

    for _ in range(WARM_UP):
        graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids)

    print("Start benchmark...")
    torch.cuda.synchronize()
    t1 = time.time()

    for _ in range(T):
        graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids)

    torch.cuda.synchronize()
    t2 = time.time()

    print("Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))

    file_path = "/home/hanshis/workspace/LongContextInfer/benchmark/report/L40_torch_new.csv"
    with open(file_path, 'a') as f:
        f.write(f"{PREFIX_LEN},{DEC_LEN},{(t2 - t1) / T},{T},{args.flash}\n")