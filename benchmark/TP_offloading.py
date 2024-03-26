import os
from re import S
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
import torch
import time

from models.cache_utils import DistributedKVCacheBuffer, DistributedSimpleCache
from models.TP_llama import distributed_init
from models.configuration_llama import LlamaConfig
from models.TP_layers import DistributedOffloadingConfig

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)
model_name_or_path = "NousResearch/Yarn-Llama-2-7b-128k"

prefill = 1024*127
gen_len = 256
on_chip_layers = 0

config = DistributedOffloadingConfig(LlamaConfig.from_pretrained(model_name_or_path), local_rank, world_size)
kv_cache =  DistributedSimpleCache(config, max_budget=prefill+3*gen_len, device=device, on_chip_layers=on_chip_layers)
kv_buffer = DistributedKVCacheBuffer(config, max_budget=prefill+3*gen_len, device=device)

# kv_cache.cpu_key_cache.normal_()
# kv_cache.cpu_value_cache.normal_()
kv_cache.seq_len = prefill

T = 5

# warm up
for _ in range(3):
    for idx in range(32):
        kv_buffer.copy_kv(kv_cache, idx)

torch.cuda.synchronize()
start = time.time()
for _ in range(T):
    for idx in range(32):
        kv_buffer.copy_kv(kv_cache, idx)
torch.cuda.synchronize()
end = time.time()
print('buffer copy kv', (end-start)/T)

# warm up
for _ in range(10):
    for idx in range(32):
        kv_buffer.copy_kv(kv_cache, idx)

torch.cuda.synchronize()
start = time.time()
for _ in range(T):
    for idx in range(32):
        kv_cache.copy_back_from_buffer(kv_buffer, idx)
torch.cuda.synchronize()
end = time.time()
print('copy back from buffer', (end-start)/T)