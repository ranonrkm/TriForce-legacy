import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from models.modeling_llama_tree import LlamaForCausalLM
from models.cache_utils import OffloadingTREESimpleCache, TREEChunkTopKCache
import torch
import time

target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
cache = OffloadingTREESimpleCache(target, 128*1024)
graph_cache = TREEChunkTopKCache(target, max_budget=4096, prefill=128*1024-256, tree_size=128, chunk_size=8)
cache.seq_len = 128*1024-256

key_ = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')
value_ = torch.randn(1, 32, 1, 128, dtype=torch.float16, device='cuda')

# warm up
time1 = time.time()
for _ in range(10):
    for i in range(32):
        key_states, value_states = cache.update(key_, value_, layer_idx=i)
torch.cuda.synchronize()
time2 = time.time()
print("[KV offloading] latency: ", (time2-time1)/10, "s")
cache.seq_len = 128*1024-256

# # offloading benchmarking
# time1 = time.time()
# for _ in range(255):
#     for i in range(32):
#         key_states, value_states = cache.update(key_, value_, layer_idx=i)
# torch.cuda.synchronize()
# time2 = time.time()
# print("[Offloading] latency: ", (time2-time1)/255, "s")


# gather_kv_incremental benchmarking
cache.seq_len = 128*1024-256
accept_list = [0,1,4,6,7,8,15,16,17,28,39,48,51,53,56,60,64,68,70,100]
time1 = time.time()
for _ in range(100):
    cache.gather_kv_incremental(accept_list, cache.seq_len)
    graph_cache.update_graph_cache(cache)
    cache.seq_len = 128*1024-256
torch.cuda.synchronize()
time2 = time.time()
print("[gather_kv_incremental] latency: ", (time2-time1)/100, "s")

input_ids = torch.randint(10, 30000, (1, 1), dtype=torch.long, device='cuda')
with torch.inference_mode():
    T = 20
    time1 = time.time()
    for _ in range(T):
        cache.seq_len = 128*1024-1
        target(input_ids=input_ids, kv_cache=cache, graph_cache=None)
    torch.cuda.synchronize()
    time2 = time.time()
    print(f"[baseline] latency: ", (time2-time1)/T, "s")


tree_size_list = [4, 8, 16, 32, 64, 128, 256, 512]

for tree_size in tree_size_list:
    path = f'/home/hanshis/workspace/Sequoia/long_tree-{tree_size}.pt'
    grow_map = torch.load(path)
    depth = grow_map['depth']
    tree_size = grow_map["size"]
    tree_mask :torch.Tensor = grow_map["mask"].cuda()
    tree_mask = (tree_mask == 0).type(torch.float16)
    tree_mask.masked_fill_(tree_mask > 0, torch.finfo(torch.float16).min)
    input_ids = torch.randint(10, 30000, (1, grow_map["size"]), dtype=torch.long, device='cuda')

    cache.seq_len = 128*1024-grow_map["size"]
    attn_mask = torch.cat([torch.zeros(tree_size, cache.seq_len, device='cuda:0'), tree_mask], dim=-1)[None, None, :, :]

    # print("input_ids: ", input_ids.shape, "attn_mask: ", attn_mask.shape)

    with torch.inference_mode():
        for _ in range(6):
            cache.seq_len = 128*1024-grow_map["size"]
            position_ids = (depth + cache.seq_len).unsqueeze(0)
            target(input_ids=input_ids, kv_cache=cache, graph_cache=None, position_ids=position_ids, attention_mask=attn_mask)
    
    with torch.inference_mode():
        T = 20
        time1 = time.time()
        for _ in range(T):
            cache.seq_len = 128*1024-grow_map["size"]
            position_ids = (depth + cache.seq_len).unsqueeze(0)
            target(input_ids=input_ids, kv_cache=cache, graph_cache=None, position_ids=position_ids, attention_mask=attn_mask)
        torch.cuda.synchronize()
        time2 = time.time()
        print(f"[verification] latency with {tree_size}: ", (time2-time1)/T, "s")