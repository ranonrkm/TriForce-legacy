from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math

####################### Batch KV cache #######################

class BatchSimpleCache:

    def __init__(self, model, max_budget=1024, bsz=2) -> None:

        self.max_budget = max_budget
        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
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
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
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