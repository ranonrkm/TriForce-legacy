from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv

def check_enough_idx(tensor, num_unique_idx=4):
    slice_unique_elements = torch.unique(tensor)
    return len(slice_unique_elements) >= num_unique_idx


def binary_search_critical_width(tensor, left, right, num_unique_idx):
    while left < right:
        mid = (left + right) // 2
        if check_enough_idx(tensor[:, :mid], num_unique_idx):
            right = mid
        else:
            left = mid + 1
    return left

def merge_topk(tensor, topk_size=4096):
    # tensor: [groups, tokens] --> [tokens]
    assert len(tensor.size()) == 2, "tensor should be 2D"
    groups = tensor.size(0)
    left = topk_size // groups
    right = topk_size
    critical_width = binary_search_critical_width(tensor, left, right, topk_size)
    x = tensor[:, :critical_width].T.flatten()
    unique, inverse, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    ret = x[index][:topk_size]
    return ret

####################### Batch KV cache #######################

class BatchSimpleCache:

    def __init__(self, model, max_budget=1024, bsz=2) -> None:

        self.max_budget = max_budget
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers
        self.bsz = bsz
        self.seq_len = torch.zeros(bsz, dtype=torch.int32).to(model.device)
        device=model.device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, bsz, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, bsz, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
    
    def print_status(self):
        print("Budget:", self.max_budget, f"| bsz: {self.bsz}", f"| Cached: {self.seq_len.cpu().numpy()}")

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int, storage_ids :torch.Tensor):

        # if layer_idx == 0:
        #     print(storage_ids, key_states.shape, value_states.shape)
        #     print("Max Budget:", self.max_budget, f"| bsz: {self.bsz}", f"| Cached: {self.seq_len}")
        indices_expanded = storage_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        self.key_cache[layer_idx].scatter_(1, indices_expanded, key_states)
        self.value_cache[layer_idx].scatter_(1, indices_expanded, value_states)

        if layer_idx == 0:
            self.seq_len += key_states.shape[-3]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.seq_len.zero_()

    def set_len(self, length):
        self.key_cache.normal_()
        self.value_cache.normal_()
        self.seq_len = length.clone()

class BatchRetrievalCache:

    def __init__(self, model, max_budget=1024, bsz=2, prefill=1024, chunk_size=8, gamma=6) -> None:

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + gamma

        self.hidden_size = model.config.hidden_size

        self.num_heads = model.config.num_key_value_heads
        self.num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        
        self.layers = model.config.num_hidden_layers
        self.bsz = bsz
        self.seq_len = torch.zeros(bsz, dtype=torch.int32).to(model.device)
        device=model.device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
    
    def print_status(self):
        print("Budget:", self.max_budget, f"| bsz: {self.bsz}", " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # key_cache: (bsz, seq_len, 32, head_dim) --> (bsz, 32, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)

        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = kv_cache.key_cache[layer_idx,:,:self.prefill].cuda().view(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_k = repeat_kv(chunk_k.permute(0, 2, 1, 3), self.num_key_value_groups) # (bsz, kv_heads, chunks, head_dim)
        
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 1, 3, 2)).squeeze(2) # (bsz, 32, chunks)

        # if self.num_key_value_groups > 1:
        #     chunk_attn = chunk_attn.reshape(self.bsz, self.num_heads, self.num_key_value_groups, self.chunks).mean(dim=2) # (bsz, kv_heads, chunks)

        # print(chunk_attn)

        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)

        if self.num_key_value_groups > 1:
            merged_results = torch.empty((self.bsz, self.num_heads, self.select_sets), dtype=torch.long, device=topk_idx.device)
            topk_idx = topk_idx.reshape(self.bsz, self.num_heads, self.num_key_value_groups, self.select_sets)
            for i in range(self.bsz):
                for j in range(self.num_heads):
                    ret = merge_topk(topk_idx[i, j], topk_size=self.select_sets) # ret: torch.Size([select_sets])
                    merged_results[i, j] = ret.clone()
            
            topk_idx = merged_results # collect ret and merge => torch.Size([bsz, num_heads, select_sets])

        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache.key_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = kv_cache.value_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int,  gamma_offset :int):
        
        self.key_cache[layer_idx][:, self.real_budget-self.gamma+gamma_offset] = key_states.clone().squeeze(1)
        self.value_cache[layer_idx][:, self.real_budget-self.gamma+gamma_offset] = value_states.clone().squeeze(1)

        return self.key_cache[layer_idx][:,:self.real_budget-self.gamma+gamma_offset+1], self.value_cache[layer_idx][:,:self.real_budget-self.gamma+gamma_offset+1]

    def update_graph_cache(self, kv_cache=None):
        # self.value_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[:,:, self.prefill:kv_cache.seq_len]
        # self.key_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[:,:, self.prefill:kv_cache.seq_len]
        bsz = kv_cache.seq_len.shape[0]
        for i in range(bsz):
            self.value_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.value_cache[:,i, self.prefill:kv_cache.seq_len[i]]
            self.key_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.key_cache[:,i, self.prefill:kv_cache.seq_len[i]]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()


########### Chain Spec Cache ############
class BatchRetrievalVerificationCache:
    def __init__(self, model, max_budget=1024, bsz=2, prefill=1024, chunk_size=8, gamma=6) -> None:
        
        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + gamma + 1
        self.bsz = bsz

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.init_graph = False

    def print_status(self):
        print("Max Budget:", self.max_budget, " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets, " | bsz:", self.bsz)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # key_cache: (bsz, seq_len, 32, head_dim) --> (bsz, 32, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)

        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        chunk_k = kv_cache.key_cache[layer_idx,:,:self.prefill].cuda().view(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)

        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache.key_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = kv_cache.value_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True

    def update_graph_cache(self, kv_cache=None):
        bsz = kv_cache.seq_len.shape[0]
        for i in range(bsz):
            self.value_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.value_cache[:,i, self.prefill:kv_cache.seq_len[i]]
            self.key_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.key_cache[:,i, self.prefill:kv_cache.seq_len[i]]

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int,  gamma_offset :int):
        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = key_states.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = value_states.clone()

        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()

class BatchStreamEvictionCache:

    def __init__(self, model, gamma=6, start_size=16, recent_size=496, bsz=2) -> None:
        
        self.bsz = bsz
        self.gamma = gamma
        self.start_size = start_size
        self.recent_size = recent_size
        self.real_budget = self.start_size + self.recent_size + self.gamma + 1 # (gamma + 1 is for spec and bonus sample, another 1 for hint)

        self.seq_len = 0 # just for prefill usage

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)

    def print_status(self):
        print("Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Real Budget:", self.real_budget, "| Cached:", self.seq_len, "| bsz:", self.bsz)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        
        incoming = key_states.shape[-3]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += incoming
        return key, value

    def spec_update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, gamma_offset=0):

        start = self.real_budget-self.gamma-1 + gamma_offset
        end = self.real_budget-self.gamma-1+new_k_cache.shape[-3] + gamma_offset

        self.key_cache[layer_idx][:, start:end] = new_k_cache.clone()
        self.value_cache[layer_idx][:, start:end] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:end], self.value_cache[layer_idx][:,:end]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict_prefill(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.key_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()
            self.value_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.value_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()

        self.seq_len = self.start_size + self.recent_size - incoming

    def evict_for_spec(self, current_seq_len):
        # current_seq_len: (bsz, )
        # self.key_cache[:,:,self.start_size:self.start_size+self.recent_size] = self.key_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
        # self.value_cache[:,:, self.start_size:self.start_size+self.recent_size] = self.value_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
        for i in range(self.bsz):
            self.key_cache[:,i,self.start_size:self.start_size+self.recent_size] = self.key_cache[:,i, current_seq_len[i]-self.recent_size:current_seq_len[i]].clone()
            self.value_cache[:,i, self.start_size:self.start_size+self.recent_size] = self.value_cache[:,i, current_seq_len[i]-self.recent_size:current_seq_len[i]].clone()



################### Distributed KV cache ####################
class DistributedBatchSimpleCache:
    def __init__(self, config, max_budget=1024, bsz=1, device=None) -> None:
        
        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.max_budget = max_budget
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        
        self.head_dim = self.hidden_size // self.config.num_attention_heads
        self.layers = self.config.num_hidden_layers
        self.bsz = bsz
        self.seq_len = torch.zeros(bsz, dtype=torch.int32).cuda()
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, bsz, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(self.device)
        self.value_cache=torch.zeros([self.layers, bsz, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(self.device)
        print(f"[Distributed Cache] Initiated for {self.local_rank}/{self.world_size} on {self.device}, shape: {self.key_cache.shape}")
    
    def print_status(self):
        print("Budget:", self.max_budget, f"| bsz: {self.bsz}", f"| Cached: {self.seq_len.cpu().numpy()}")

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int, storage_ids :torch.Tensor):
        indices_expanded = storage_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        self.key_cache[layer_idx].scatter_(1, indices_expanded, key_states)
        self.value_cache[layer_idx].scatter_(1, indices_expanded, value_states)

        if layer_idx == 0:
            self.seq_len += key_states.shape[-3]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.seq_len.zero_()
        print(f"[Distributed Cache] Reset for {self.local_rank}/{self.world_size} on {self.device}, shape: {self.key_cache.shape}")

    def set_len(self, length):
        self.key_cache.normal_()
        self.value_cache.normal_()
        self.seq_len = length.clone()

    def copy_back_from_buffer(self, kv_buffer, layer_idx:int):

        storage_ids = torch.cat([torch.arange(start, end) for start, end in zip(self.seq_len, kv_buffer.seq_len)]).unsqueeze(0) # bsz==1
        indices_expanded = storage_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        self.key_cache[layer_idx].scatter_(1, indices_expanded, kv_buffer.key_cache.gather(1, indices_expanded))
        self.value_cache[layer_idx].scatter_(1, indices_expanded, kv_buffer.value_cache.gather(1, indices_expanded))

        if layer_idx == self.layers - 1:
            self.seq_len = kv_buffer.seq_len.clone()

class DistributedBatchKVCacheBuffer:
    def __init__(self, config, max_budget=1024, bsz=1, device=None) -> None:

        self.config = config
        self.max_budget = max_budget
        self.device = device
        self.dtype = torch.float16
        self.bsz = bsz

        self.world_size = config.world_size
        self.local_rank = config.local_rank

        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_key_value_heads // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.key_cache = torch.zeros(bsz, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.value_cache = torch.zeros(bsz, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.seq_len = torch.zeros(bsz, dtype=torch.int32).to(self.device)

    def copy_kv(self, kv_cache, layer_idx):
        self.key_cache.copy_(kv_cache.key_cache[layer_idx], non_blocking=True)
        self.value_cache.copy_(kv_cache.value_cache[layer_idx], non_blocking=True)
        self.seq_len = kv_cache.seq_len.clone()

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int, storage_ids :torch.Tensor):
        indices_expanded = storage_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        self.key_cache.scatter_(1, indices_expanded, key_states)
        self.value_cache.scatter_(1, indices_expanded, value_states)

        if layer_idx == 0:
            self.seq_len += key_states.shape[-3]
        return self.key_cache, self.value_cache

class DistributedBatchRetrievalCache:

    def __init__(self, config, max_budget=1024, bsz=1, device=None, prefill=1024, chunk_size=8, gamma=6) -> None:

        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.layers = self.config.num_hidden_layers
        self.bsz = bsz

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + gamma
        
        self.device=device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, bsz, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        print(f"[Distributed Retrieval Cache] Initiated for {self.local_rank+1}/{self.world_size} on {self.device}, shape: {self.key_cache.shape}")

    def print_status(self):
        print("Budget:", self.max_budget, f"| bsz: {self.bsz}", " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # key_cache: (bsz, seq_len, 32, head_dim) --> (bsz, 32, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)

        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = kv_cache.key_cache[layer_idx,:,:self.prefill].cuda().view(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_k = repeat_kv(chunk_k.permute(0, 2, 1, 3), self.num_key_value_groups) # (bsz, kv_heads, chunks, head_dim)
        
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 1, 3, 2)).squeeze(2) # (bsz, 32, chunks)

        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)

        if self.num_key_value_groups > 1:
            merged_results = torch.empty((self.bsz, self.num_heads, self.select_sets), dtype=torch.long, device=topk_idx.device)
            topk_idx = topk_idx.reshape(self.bsz, self.num_heads, self.num_key_value_groups, self.select_sets)
            for i in range(self.bsz):
                for j in range(self.num_heads):
                    ret = merge_topk(topk_idx[i, j], topk_size=self.select_sets) # ret: torch.Size([select_sets])
                    merged_results[i, j] = ret.clone()
            
            topk_idx = merged_results # collect ret and merge => torch.Size([bsz, num_heads, select_sets])

        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache.key_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = kv_cache.value_cache[layer_idx][:, :self.prefill].reshape(self.bsz, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(self.bsz, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True
            print(f"[Distributed Retrieval Cache] Built for {self.local_rank+1}/{self.world_size} on {self.device}, shape: {self.key_cache.shape}")

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int,  gamma_offset :int):
        
        self.key_cache[layer_idx][:, self.real_budget-self.gamma+gamma_offset] = key_states.squeeze(1)
        self.value_cache[layer_idx][:, self.real_budget-self.gamma+gamma_offset] = value_states.squeeze(1)

        return self.key_cache[layer_idx][:,:self.real_budget-self.gamma+gamma_offset+1], self.value_cache[layer_idx][:,:self.real_budget-self.gamma+gamma_offset+1]

    def update_graph_cache(self, kv_cache=None):
        bsz = kv_cache.seq_len.shape[0]
        for i in range(bsz):
            self.value_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.value_cache[:,i, self.prefill:kv_cache.seq_len[i]]
            self.key_cache[:,i,self.max_budget-(kv_cache.seq_len[i]-self.prefill):self.max_budget] = kv_cache.key_cache[:,i, self.prefill:kv_cache.seq_len[i]]

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        print(f"[Distributed Retrieval Cache] Reset for {self.local_rank+1}/{self.world_size} on {self.device}, shape: {self.key_cache.shape}")

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()