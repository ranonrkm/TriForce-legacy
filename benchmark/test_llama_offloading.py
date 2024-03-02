import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from models.modeling_llama import LlamaForCausalLM
from models.cache_utils import OffloadingFlashSimpleCache
import torch
import time

target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
cache = OffloadingFlashSimpleCache(target, 128*1024)
cache.seq_len = 128*1024-256

key_ = torch.randn(1, 1, 32, 128, dtype=torch.float16, device='cuda')
value_ = torch.randn(1, 1, 32, 128, dtype=torch.float16, device='cuda')


# warm up
time1 = time.time()
for _ in range(10):
    for i in range(32):
        key_states, value_states = cache.update(key_, value_, layer_idx=i)
torch.cuda.synchronize()
time2 = time.time()
print("[Warm up] latency: ", (time2-time1)/10, "s")
cache.seq_len = 128*1024-256

# offloading benchmarking
time1 = time.time()
for _ in range(255):
    for i in range(32):
        key_states, value_states = cache.update(key_, value_, layer_idx=i)
torch.cuda.synchronize()
time2 = time.time()
print("[Offloading] latency: ", (time2-time1)/255, "s")