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