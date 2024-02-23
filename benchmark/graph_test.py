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
from models.modeling_llama import LlamaForCausalLM, LlamaConfig
from models.cache_utils import SimpleCache, FlashSimpleCache, GraphFlashSimpleCache, GraphFlashStreamLLMCache
from utils.graph_infer import GraphInferenceEngine

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=1000, help='time')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--flash', action='store_true', help='flash')
parser.add_argument('--model_name', type=str, default="NousResearch/Yarn-Llama-2-7b-128k", help='model name')
args = parser.parse_args()

PREFIX_LEN = args.P
T = args.T
WARM_UP = 10

host = socket.gethostname()
if 'lovelace' in host:
    file_path = "/home/hanshis/workspace/LongContextInfer/benchmark/report/L40_llama_7B_128K_graph.csv"
else:
    file_path = "/data/home/beidic/hanshi/LongContextInfer/benchmark/report/A100_llama_7B_128K_graph.csv"

try:
    with open(file_path, 'r') as f:
        contents = f.read()
except FileNotFoundError:
    contents = ""

if not contents:
    with open(file_path, 'a') as f:
        f.write("model,prefill,len,latency,repeat_time,flash\n")

if args.model_name != "4bit":
    model_name = args.model_name
    config = LlamaConfig.from_pretrained(model_name)
    # if args.flash:
    config.flash = True
    if config.max_position_embeddings < 4096:
        config.max_position_embeddings = PREFIX_LEN + 1
    model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, device_map="auto")
else:
    model_name="TheBloke/Yarn-Llama-2-7B-128K-GPTQ"
    model = LlamaForCausalLM.from_pretrained("TheBloke/Yarn-Llama-2-7B-128K-GPTQ", revision="gptq-4bit-128g-actorder_True", device_map="auto")

DEC_LEN = 1
MAX_LEN = PREFIX_LEN + 1

cache = FlashSimpleCache(model, MAX_LEN)
graph_cache = GraphFlashStreamLLMCache(model, max_budget=MAX_LEN, prefill=PREFIX_LEN, gen_len=DEC_LEN)

cache.reset()
graph_cache.reset()

prefix = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=model.device)
assert prefix.shape[-1] == PREFIX_LEN

graph_engine = GraphInferenceEngine(model, cache, graph_cache)
graph_engine.initialize_cuda_graph([DEC_LEN])

graph_engine.inference(input_ids=prefix)

graph_cache.init_stream_cache(kv_cache=cache)

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=model.device)
storage_ids = torch.arange(DEC_LEN, device=model.device) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)

for _ in range(WARM_UP):
    graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids)

graph_cache.update_stream_cache(kv_cache=cache)

print("Start benchmark...")
torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    graph_engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids)
torch.cuda.synchronize()
t2 = time.time()

print("Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))