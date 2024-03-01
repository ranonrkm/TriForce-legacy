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
from models.cache_utils import SimpleCache, FlashSimpleCache, GraphFlashSimpleCache, GraphSimpleCache, GraphFlashStreamLLMVerificationCache, GraphFlashStreamEvictionCache, GraphFlashChunkTopKVerificationCache, GraphFlashStreamEvictionCache_V2
from models.modeling_llama_68m_v2 import LlamaForCausalLM as LlamaForCausalLM_68M
from utils.chain_infer import GraphInferenceEngine

from utils.sampling import norm_logits, sample

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=2000, help='time')
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
    # model_name = args.model_name
    # config = LlamaConfig.from_pretrained(model_name)
    # # if args.flash:
    # config.flash = True
    # if config.max_position_embeddings < 4096:
    #     config.max_position_embeddings = PREFIX_LEN + 1
    # model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, device_map="auto")
    model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
else:
    model_name="TheBloke/Yarn-Llama-2-7B-128K-GPTQ"
    model = LlamaForCausalLM.from_pretrained("TheBloke/Yarn-Llama-2-7B-128K-GPTQ", revision="gptq-4bit-128g-actorder_True", device_map="auto")

# print(model)
# sleep(1000)

# DEC_LEN_LIST = [1,2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512]

DEC_LEN_LIST = [1]
gamma=5
MAX_LEN = PREFIX_LEN + 1

draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="cuda:0")

chunk_size = 8
max_budget = int(0.05 * PREFIX_LEN) // chunk_size * chunk_size
cache = FlashSimpleCache(model, MAX_LEN+100)
graph_cache = GraphFlashChunkTopKVerificationCache(model, max_budget=max_budget, prefill=PREFIX_LEN, gamma=gamma, chunk_size=8)
draft_cache = GraphFlashStreamEvictionCache_V2(draft, start_size=16, recent_size=250-16, gamma=gamma)

graph_engine = GraphInferenceEngine(model, cache, graph_cache, draft, draft_cache)
graph_engine.initialize_cuda_graph(5, probs=True)

def test_real_draft(gamma_offset, pred_token_idx):
    speculation_probs = []
    generated_ids = []
    return_generated_ids.append([[1]])
    n=0
    
    storage_ids = torch.tensor([graph_engine.engine.draft_cache.start_size + graph_engine.engine.draft_cache.recent_size + gamma_offset], device=graph_engine.engine.draft.device)
    # position_ids = storage_ids.clone().unsqueeze(0)
    # print(storage_ids, position_ids, gamma_offset)

    probs = graph_engine.graph_draft_inference(input_ids=pred_token_idx, gamma_offset = gamma_offset)

    # probs = norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
    pred_token_idx = sample(probs)
    
    speculation_probs.append(probs[0])
    generated_ids.append(pred_token_idx.item())

def test_real_verify(storage_ids, position_ids):
    # storage_ids = torch.arange(graph_engine.engine.graph_cache.max_budget, graph_engine.engine.graph_cache.max_budget+gamma+1, device=graph_engine.engine.model.device)
    position_ids = torch.arange(graph_engine.engine.kv_cache.seq_len, graph_engine.engine.kv_cache.seq_len+gamma+1, device=graph_engine.engine.model.device).unsqueeze(0)

    probs = graph_engine.graph_verify(input_ids=verify_tokens, storage_ids=storage_ids, position_ids=position_ids)

    pred_token_idx = sample(probs[-1])

    count = 0
    verify_probs = []


for DEC_LEN in DEC_LEN_LIST:
    cache.reset()
    graph_cache.reset()
    prefix = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=model.device)
    assert prefix.shape[-1] == PREFIX_LEN

    input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=model.device)
    storage_ids = torch.arange(DEC_LEN, device=model.device) + PREFIX_LEN
    position_ids = storage_ids.clone().unsqueeze(0)
    for _ in range(WARM_UP):
        graph_engine.graph_draft_inference(input_ids=input_ids, gamma_offset=0)

    print("Start benchmark...")
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        graph_engine.graph_draft_inference(input_ids=input_ids, gamma_offset=0)
    torch.cuda.synchronize()
    t2 = time.time()

    print("[Draft Run] Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))

    temperature = 0.6
    top_k = -1
    top_p = 0.9

    pred_token_idx = torch.randint(low=3, high=30000, size=(1, 1), device=model.device)
    draft_count = 0
    speculation_probs = []
    generated_ids = []
    return_generated_ids = []
    return_speculation_probs = []

    for _ in range(WARM_UP):
        test_real_draft(0, pred_token_idx)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        test_real_draft(0, pred_token_idx)
    torch.cuda.synchronize()
    t2 = time.time()

    print("[Real Draft] Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))


    verify_tokens = torch.full((1, gamma + 1), 100, device=graph_engine.engine.model.device)
    storage_ids = torch.arange(graph_engine.engine.graph_cache.max_budget, graph_engine.engine.graph_cache.max_budget+gamma+1, device=graph_engine.engine.model.device)
    position_ids = torch.arange(graph_engine.engine.kv_cache.seq_len, graph_engine.engine.kv_cache.seq_len+gamma+1, device=graph_engine.engine.model.device).unsqueeze(0)

    for _ in range(WARM_UP):
        graph_engine.graph_verify(input_ids=verify_tokens, storage_ids=storage_ids, position_ids=position_ids)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        graph_engine.graph_verify(input_ids=verify_tokens, storage_ids=storage_ids, position_ids=position_ids)
    torch.cuda.synchronize()
    t2 = time.time()

    print("[Verify] Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))


    for _ in range(WARM_UP):
        test_real_verify(storage_ids, position_ids)
    
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        test_real_verify(storage_ids, position_ids)
    torch.cuda.synchronize()
    t2 = time.time()

    print("[Real Verify] Prefix Length :{}, Decode Length :{}, inference time:{}s".format(PREFIX_LEN, DEC_LEN, (t2 - t1)/ T))