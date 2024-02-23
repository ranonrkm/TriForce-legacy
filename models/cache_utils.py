from typing import Any, Dict, List, Optional, Tuple
import torch
import math

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

class SimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            if hasattr(model, 'gpt_neox'):
                device = model.gpt_neox.layers[i].attention.query_key_value.weight.device
                dtype = model.gpt_neox.layers[i].attention.query_key_value.weight.dtype
            else:
                device = model.device
                dtype = torch.float16
                # device = model.model.layers[i].self_attn.q_proj.weight.device
                # dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]

        return key, value

    def tree_rollback(self, flag):
        # flag is 1 or 2
        if flag == 1:
            return
        else:
            for layer in range(self.layers):
                self.key_cache[layer][:,:,self.seq_len-2:self.seq_len-1] = self.key_cache[layer][:,:,self.seq_len-1:self.seq_len]
                self.value_cache[layer][:,:,self.seq_len-2:self.seq_len-1] = self.value_cache[layer][:,:,self.seq_len-1:self.seq_len]

class FlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        # self.key_cache: List[torch.Tensor] = []
        # self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)

        # for i in range(self.layers):
        #     if hasattr(model, 'gpt_neox'):
        #         device = model.gpt_neox.layers[i].attention.query_key_value.weight.device
        #         dtype = model.gpt_neox.layers[i].attention.query_key_value.weight.dtype
        #     else:
        #         # device = model.model.layers[i].self_attn.q_proj.weight.device
        #         # dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
        #         device = model.device
        #         dtype = torch.float16
        #     # print(device, dtype)
        #     self.key_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
        #     self.value_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class StreamLLMCache(Cache):
    def __init__(self, model, max_budget=1024, start_size=4, recent_size=16, skip_start_layers=-1, gamma=4) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        ### for faster speed ###
        self.cached_key: List[torch.Tensor] = []
        self.cached_value: List[torch.Tensor] = []
        
        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size
        self.max_budget = max_budget
        self.skip_start_layers = skip_start_layers
        self.gamma = gamma
        self.spec_time = 0
        self.cache_kv_size = self.start_size+self.recent_size+self.gamma

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))

            self.cached_key.append(torch.zeros([1, self.num_heads, self.start_size+self.recent_size+self.gamma, self.head_dim], dtype=dtype).to(device))
            self.cached_value.append(torch.zeros([1, self.num_heads, self.start_size+self.recent_size+self.gamma, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Skip Start Layers:", self.skip_start_layers)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

            self.cached_key[i].zero_()
            self.cached_value[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache

        # print(self.key_cache[layer_idx].shape, self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]].shape, key_states.shape)

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]
            self.spec_time = 0

        return key, value

    def update_cache(self):
        for layer in range(self.layers):
            self.cached_key[layer] = torch.cat([
                self.key_cache[layer][:,:, :self.start_size], 
                self.key_cache[layer][:,:, -self.recent_size + self.seq_len:self.seq_len + self.gamma]
            ], dim=-2)

            self.cached_value[layer] = torch.cat([
                self.value_cache[layer][:,:, :self.start_size], 
                self.value_cache[layer][:,:, -self.recent_size + self.seq_len:self.seq_len + self.gamma]
            ], dim=-2)

    def speculation_update(self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
    ):
        # Update the cache
        assert key_states.shape[-2] == 1

        if self.seq_len <= self.start_size + self.recent_size:
            return self.update(key_states, value_states, layer_idx)

        # assert self.cached_key[layer_idx].shape[-2] == self.cache_kv_size
        self.cached_key[layer_idx][:, :, self.cache_kv_size-self.gamma + self.spec_time:self.cache_kv_size-self.gamma + self.spec_time+1] = key_states

        self.cached_value[layer_idx][:, :, self.cache_kv_size-self.gamma + self.spec_time:self.cache_kv_size-self.gamma + self.spec_time+1] = value_states

        # self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = key_states
        # self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = value_states

        # key = torch.cat([
        #     self.key_cache[layer_idx][:, :, :self.start_size], 
        #     self.key_cache[layer_idx][:, :, -self.recent_size + self.seq_len:self.seq_len + 1]
        # ], dim=-2)
        # value = torch.cat([
        #     self.value_cache[layer_idx][:, :, :self.start_size], 
        #     self.value_cache[layer_idx][:, :, -self.recent_size + self.seq_len:self.seq_len + 1]
        # ], dim=-2)
        ###!!! test all close

        if layer_idx == self.layers-1:
            self.seq_len += 1
            self.spec_time += 1

        return self.cached_key[layer_idx][:,:,:self.cache_kv_size-self.gamma + self.spec_time+1], self.cached_value[layer_idx][:,:,:self.cache_kv_size-self.gamma + self.spec_time+1]

    def tree_rollback(self, flag):
        # flag is 1 or 2
        if flag == 1:
            return
        else:
            for layer in range(self.layers):
                self.cached_key[layer][:,:,self.seq_len-2:self.seq_len-1] = self.cached_key[layer][:,:,self.seq_len-1:self.seq_len]
                self.cached_value[layer][:,:,self.seq_len-2:self.seq_len-1] = self.cached_value[layer][:,:,self.seq_len-1:self.seq_len]

class FlashStreamLLMCache(Cache):
    def __init__(self, model, max_budget=1024, start_size=4, recent_size=16, skip_start_layers=-1, gamma=4) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        ### for faster speed ###
        self.cached_key: List[torch.Tensor] = []
        self.cached_value: List[torch.Tensor] = []

        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size
        self.max_budget = max_budget
        self.skip_start_layers = skip_start_layers
        self.gamma = gamma
        self.spec_time = 0
        self.cache_kv_size = self.start_size+self.recent_size+self.gamma

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))

            self.cached_key.append(torch.zeros([1, self.start_size+self.recent_size+self.gamma, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.cached_value.append(torch.zeros([1, self.start_size+self.recent_size+self.gamma, self.num_heads, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Skip Start Layers:", self.skip_start_layers)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

            self.cached_key[i].zero_()
            self.cached_value[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache

        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]
            self.spec_time = 0

        return key, value

    def update_cache(self):
        for layer in range(self.layers):
            self.cached_key[layer] = torch.cat([
                self.key_cache[layer][:, :self.start_size], 
                self.key_cache[layer][:, -self.recent_size + self.seq_len:self.seq_len + self.gamma]
            ], dim=-3)

            self.cached_value[layer] = torch.cat([
                self.value_cache[layer][:, :self.start_size], 
                self.value_cache[layer][:, -self.recent_size + self.seq_len:self.seq_len + self.gamma]
            ], dim=-3)

    def speculation_update(self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
    ):
        # Update the cache
        # assert key_states.shape[-3] == 1

        if self.seq_len <= self.start_size + self.recent_size:
            return self.update(key_states, value_states, layer_idx)

        self.cached_key[layer_idx][:, self.cache_kv_size-self.gamma + self.spec_time:self.cache_kv_size-self.gamma + self.spec_time+key_states.shape[-3]] = key_states

        self.cached_value[layer_idx][:, self.cache_kv_size-self.gamma + self.spec_time:self.cache_kv_size-self.gamma + self.spec_time+key_states.shape[-3]] = value_states

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]
            self.spec_time += 1

        return self.cached_key[layer_idx][:,:self.cache_kv_size-self.gamma + self.spec_time+key_states.shape[-3]], self.cached_value[layer_idx][:,:self.cache_kv_size-self.gamma + self.spec_time+key_states.shape[-3]]

class FlashEvictStreamLLMCache(Cache):
    def __init__(self, model, start_size=4, recent_size=256) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, start_size+recent_size, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, start_size+recent_size, self.num_heads, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, self.start_size:-incoming] = self.key_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()
            self.value_cache[layer_idx][:, self.start_size:-incoming] = self.value_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()

        self.seq_len = self.start_size + self.recent_size - incoming

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # evict kv cache

        incoming = key_states.shape[-3]

        assert self.seq_len + incoming <= self.start_size + self.recent_size

        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + incoming] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + incoming] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]
        return key, value

class EvictStreamLLMCache(Cache):
    def __init__(self, model, start_size=4, recent_size=256) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, start_size+recent_size, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, start_size+recent_size, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, :, self.start_size:-incoming] = self.key_cache[layer_idx][:, :, self.seq_len-size_keep:self.seq_len]
            self.value_cache[layer_idx][:, :, self.start_size:-incoming] = self.value_cache[layer_idx][:, :, self.seq_len-size_keep:self.seq_len]

        self.seq_len = self.start_size + self.recent_size - incoming

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # evict kv cache

        incoming = key_states.shape[-2]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + incoming] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + incoming] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]
        return key, value

class EvictH2OCache(Cache):
    def __init__(self, model, start_size=128, recent_size=128) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.hh_score: List[torch.Tensor] = []
        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.q_num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, start_size+recent_size, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, start_size+recent_size, self.head_dim], dtype=dtype).to(device))
            self.hh_score.append(torch.zeros([1, self.num_heads, start_size+recent_size], dtype=torch.float32).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
            self.hh_score[i].zero_()

    def evict(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = (self.seq_len - self.recent_size) - (incoming - (self.start_size + self.recent_size - self.seq_len))
            selected_hh_scores = self.hh_score[layer_idx][:, :, :self.seq_len-self.recent_size]

            # print(selected_hh_scores, self.hh_score[layer_idx].shape,selected_hh_scores.shape, size_keep)
            # exit()
            # print(size_keep, selected_hh_scores.shape, self.hh_score[layer_idx].shape)
            _, keep_topk = torch.topk(selected_hh_scores, k=size_keep, dim=-1)
            keep_topk = keep_topk.sort().values

            mask = torch.zeros(selected_hh_scores.shape, dtype=torch.bool, device=self.hh_score[layer_idx].device)
            mask = mask.scatter(-1, keep_topk, 1)

            ##### evict kv cache #####
            self.key_cache[layer_idx][:, :, :size_keep] = self.key_cache[layer_idx][:, :, :self.seq_len-self.recent_size][mask].view(1, self.num_heads, -1, self.head_dim)
            self.value_cache[layer_idx][:, :, :size_keep] = self.value_cache[layer_idx][:, :, :self.seq_len-self.recent_size][mask].view(1, self.num_heads, -1, self.head_dim)

            self.key_cache[layer_idx][:, :, size_keep:-incoming] = self.key_cache[layer_idx][:, :, self.seq_len-self.recent_size:self.seq_len]
            self.value_cache[layer_idx][:, :, size_keep:-incoming] = self.value_cache[layer_idx][:, :, self.seq_len-self.recent_size:self.seq_len]
            
            ##### evict hh score #####
            self.hh_score[layer_idx][:, :, :size_keep] = self.hh_score[layer_idx][:, :, :self.seq_len-self.recent_size][mask].view(1, self.num_heads, -1)
            self.hh_score[layer_idx][:, :, size_keep:-incoming] = self.hh_score[layer_idx][:, :, self.seq_len-self.recent_size:self.seq_len]
            self.hh_score[layer_idx][:, :, -incoming:] = 0


        self.seq_len = self.start_size + self.recent_size - incoming

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # evict kv cache

        incoming = key_states.shape[-2]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + incoming] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + incoming] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]
        return key, value
    
    def update_hh_score(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        attn_weights = attn_weights.sum(2)

        cum = self.q_num_heads // self.num_heads
        for i in range(self.num_heads): # q_head 32, k_v_head 4
            self.hh_score[layer_idx][:, i, :attn_weights.shape[-1]] += torch.sum(attn_weights[:, i*cum:(i+1)*cum, :], axis=1)

class H2OCache(Cache):
    def __init__(self, model, max_budget=1024, heavy_size=16, recent_size=16, skip_start_layers=-1) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.max_budget = max_budget
        self.skip_start_layers = skip_start_layers
        self.hh_score: List[torch.Tensor] = []

        self.heavy_key_cache: List[torch.Tensor] = []
        self.heavy_value_cache: List[torch.Tensor] = []

        self.speculation_update_times = 0

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.q_num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(self.layers):
            if hasattr(model.model.layers[i].self_attn.q_proj, 'weight'):
                device = model.model.layers[i].self_attn.q_proj.weight.device
                dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            else:
                device = model.model.layers[i].self_attn.q_proj.qweight.device
                dtype = torch.float16
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.hh_score.append(torch.zeros([1, self.num_heads, self.max_budget], dtype=torch.float32).to(device))

            self.heavy_key_cache.append(torch.zeros([1, self.num_heads, self.heavy_size, self.head_dim], dtype=dtype).to(device))
            self.heavy_value_cache.append(torch.zeros([1, self.num_heads, self.heavy_size, self.head_dim], dtype=dtype).to(device))
    
    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| Heavy Size:", self.heavy_size, "| Recent Size:", self.recent_size, "| Single Layer HH Score:", self.hh_score[0].shape)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
            self.hh_score[i].zero_()
            self.heavy_key_cache[i].zero_()
            self.heavy_value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]
            self.speculation_update_times = 0

        return key, value
    
    def update_hh_score(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> None:
            
        attn_weights = attn_weights.sum(2)
        if layer_idx == self.layers-1:
            assert attn_weights.shape[-1] == self.seq_len, (attn_weights.shape, self.seq_len)


        if self.q_num_heads != self.num_heads:
            cum = self.q_num_heads // self.num_heads
            for i in range(self.num_heads): # q_head 32, k_v_head 4
                self.hh_score[layer_idx][:, i, :attn_weights.shape[-1]] += torch.sum(attn_weights[:, i*cum:(i+1)*cum, :], axis=1) / cum

        else:
            # print(attn_weights.shape, self.hh_score[layer_idx][:, :, :attn_weights.shape[-1]].shape)
            self.hh_score[layer_idx][:, :, :attn_weights.shape[-1]] += attn_weights


    def update_cache(self):
        # update heavy cache
        for layer_idx in range(self.layers):
            selected_hh_scores = self.hh_score[layer_idx][:, :, :self.seq_len - self.recent_size]#  / torch.arange(self.seq_len - self.recent_size, 0, -1, dtype=torch.long, device=self.device).repeat(32, 1)

            _, keep_topk = torch.topk(selected_hh_scores, k=self.heavy_size, dim=-1)
            keep_topk = keep_topk.sort().values

            mask = torch.zeros(selected_hh_scores.shape, dtype=torch.bool, device=self.hh_score[layer_idx].device)
            mask = mask.scatter(-1, keep_topk, 1)

            self.heavy_key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :self.seq_len - self.recent_size][mask].view(1, self.num_heads, -1, self.head_dim)
            self.heavy_value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :self.seq_len - self.recent_size][mask].view(1, self.num_heads, -1, self.head_dim)

    def speculation_update(self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Update the cache
        assert key_states.shape[-2] == 1

        if self.seq_len <= self.heavy_size + self.recent_size:
            return self.update(key_states, value_states, layer_idx, cache_kwargs)

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = value_states
        
        ## [heavy, recent + 1 (new coming, not in cache before)]
        key = torch.cat([
            self.heavy_key_cache[layer_idx],
            self.key_cache[layer_idx][:, :, -self.recent_size + self.seq_len - self.speculation_update_times:self.seq_len + 1]
        ], dim=-2)

        value = torch.cat([
            self.heavy_value_cache[layer_idx],
            self.value_cache[layer_idx][:, :, -self.recent_size + self.seq_len - self.speculation_update_times:self.seq_len + 1]
        ], dim=-2)

        if layer_idx == self.layers-1:
            self.seq_len += 1
            self.speculation_update_times += 1

        return key, value

class DejaVuCache(Cache):
    def __init__(self, model, max_budget=1024, topk_size=16, skip_start_layers=-1) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.topk_size = topk_size
        self.max_budget = max_budget
        self.skip_start_layers = skip_start_layers

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        # initialize empty kv cache using max_budget
        for i in range(32):
            device = model.model.layers[i].self_attn.q_proj.weight.device
            dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]
            self.speculation_update_times = 0

        return key, value
    
    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| TopK Size:", self.topk_size)

    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def speculation_update(self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        query_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # Update the cache
        assert key_states.shape[-2] == 1

        if self.seq_len <= self.topk_size:
            return self.update(key_states, value_states, layer_idx)

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = value_states

        # fake simulation
        attn_weights = torch.matmul(query_states, self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]].transpose(2, 3)) / math.sqrt(self.head_dim) # (bsz, 32, 1, kv_seq_len)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        assert attn_weights.shape[2] == 1
        attn_weights = attn_weights.squeeze(2) # (bsz, 32, kv_seq_len)
        _, topk_idx = torch.topk(attn_weights, k=self.topk_size, dim=-1)
        # print(topk_idx)
        topk_idx = topk_idx.sort().values

        mask = torch.zeros(attn_weights.shape, dtype=torch.bool, device=attn_weights.device)
        mask = mask.scatter(-1, topk_idx, 1)

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]][mask].view(1, self.num_heads, -1, self.head_dim)
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]][mask].view(1, self.num_heads, -1, self.head_dim)

        assert key.shape[-2] == self.topk_size
        assert value.shape[-2] == self.topk_size

        if layer_idx == self.layers-1:
            self.seq_len += 1

        return key, value

class ChunkCache(Cache):
    def __init__(self, model, max_budget=1024, chunk_size=256, budget=0.1, skip_start_layers=-1, prefill=1024) -> None:
        

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.chunk_k: List[torch.Tensor] = []

        self.seq_len = 0
        self.max_budget = max_budget

        self.prefill = prefill
        self.skip_start_layers = skip_start_layers
        self.chunk_size = chunk_size
        self.chunks = prefill // chunk_size
        self.select_sets = int(budget * self.chunks)
        # check prefill is multiple of chunk_size
        assert prefill % chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {chunk_size}"
        assert self.select_sets <= self.chunks, f"select_sets should be less than chunks, got {self.select_sets} > {self.chunks}"
        assert chunk_size * self.select_sets <= max_budget, f"chunk_size * select_sets should be less than max_budget, got {chunk_size} * {self.select_sets} > {max_budget}"
        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            if hasattr(model, 'gpt_neox'):
                device = model.gpt_neox.layers[i].attention.query_key_value.weight.device
                dtype = model.gpt_neox.layers[i].attention.query_key_value.weight.dtype
            else:
                device = model.model.layers[i].self_attn.q_proj.weight.device
                dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))

            self.chunk_k.append(torch.zeros([1, self.num_heads, self.chunks, self.head_dim], dtype=dtype).to(device))

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| Chunk Size:", self.chunk_size, "| Select Sets:", self.select_sets, "| Chunks:", self.chunks, "| Skip Start Layers:", self.skip_start_layers, "| PreFill:", self.prefill, " | ratio:", self.select_sets/self.chunks)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
            self.chunk_k[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]

        return key, value

    def update_chunk_k(self):
        for layer in range(self.layers):
            self.chunk_k[layer] = self.key_cache[layer][:, :, :self.prefill].view(1, self.num_heads, self.chunks, self.chunk_size, self.head_dim).mean(dim=-2)

    def speculation_update(self, key_states, value_states, layer_idx, query_states):
        # Update the cache
        # key_states: (bsz, 32, 1, head_dim)
        # value_states: (bsz, 32, 1, head_dim)
        # query_states: (bsz, 32, 1, head_dim)
        # chunk_k: (bsz, 32, chunks, head_dim)

        if self.seq_len <= self.chunk_size * self.select_sets:
            return self.update(key_states, value_states, layer_idx)

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = value_states

        assert key_states.shape[-2] == 1
        assert query_states.shape[-2] == 1, "query_states should be 1 for spec update"

        chunk_attn = torch.matmul(query_states, self.chunk_k[layer_idx].transpose(2, 3)) / math.sqrt(self.head_dim) # (bsz, 32, 1, chunks)
        chunk_attn = chunk_attn.squeeze(2)
        # select topk self.select_sets

        # Not include sink cache
        # _, topk_idx = torch.topk(chunk_attn, k=self.select_sets, dim=-1)
        # topk_idx = topk_idx.sort().values


        # include sink cache
        #!!! NEED TO FIX
        chunk_attn_sliced = chunk_attn[:, :, 1:]
        _, topk_idx_sliced = torch.topk(chunk_attn_sliced, k=self.select_sets-1, dim=-1)
        topk_idx_sliced_adjusted = topk_idx_sliced + 1
        topk_idx_sorted = topk_idx_sliced_adjusted.sort().values
        initial_idx = torch.zeros((1, self.num_heads, 1), dtype=torch.long, device=topk_idx_sorted.device)
        topk_idx = torch.cat((initial_idx, topk_idx_sorted), dim=-1)


        # key = torch.empty((1, self.num_heads, self.chunk_size*self.select_sets + self.seq_len + 1 - self.prefill, self.head_dim), device=key_states.device, dtype=key_states.dtype)
        # value = torch.empty((1, self.num_heads, self.chunk_size*self.select_sets + self.seq_len + 1 - self.prefill, self.head_dim), device=value_states.device, dtype=value_states.dtype)
        # for i in range(topk_idx.shape[1]):
        #     for j in range(topk_idx.shape[2]):
        #         idx = topk_idx[0, i, j]
        #         start = idx * self.chunk_size
        #         end = start + self.chunk_size
        #         key[:, i, j*self.chunk_size:(j+1)*self.chunk_size] = self.key_cache[layer_idx][:, i, start:end]
        #         value[:, i, j*self.chunk_size:(j+1)*self.chunk_size] = self.value_cache[layer_idx][:, i, start:end]
        
        # key[:, :, self.chunk_size*self.select_sets:] = self.key_cache[layer_idx][:, :, self.prefill:self.seq_len + 1]
        # value[:, :, self.chunk_size*self.select_sets:] = self.value_cache[layer_idx][:, :, self.prefill:self.seq_len + 1]

        key_reshape = self.key_cache[layer_idx][:, :, :self.prefill].reshape(1, self.num_heads, self.chunks, self.head_dim*self.chunk_size)
        value_reshape = self.value_cache[layer_idx][:, :, :self.prefill].reshape(1, self.num_heads, self.chunks, self.head_dim*self.chunk_size)

        index_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # print(topk_idx)
        # print(index_expanded)

        # exit()

        result_flattened = torch.gather(key_reshape, 2, index_expanded.reshape(1, self.num_heads, self.select_sets, -1))
        key = result_flattened.view(1, self.num_heads, self.select_sets*self.chunk_size, self.head_dim)

        result_flattened = torch.gather(value_reshape, 2, index_expanded.reshape(1, self.num_heads, self.select_sets, -1))
        value = result_flattened.view(1, self.num_heads, self.select_sets*self.chunk_size, self.head_dim)

        key = torch.cat([key, self.key_cache[layer_idx][:, :, self.prefill:self.seq_len + 1]], dim=-2)
        value = torch.cat([value, self.value_cache[layer_idx][:, :, self.prefill:self.seq_len + 1]], dim=-2)


        ##### for debug #####
        # if layer_idx == 2:
        #     self.fake_k = key
        #     self.fake_v = value
        #     self.topk_idx = topk_idx
        #     self.query_states = query_states

        # print(key.shape)

        # exit()
        #### Maybe Use Cache mechanism to update the cache (like cpu)

        if layer_idx == self.layers-1:
            self.seq_len += 1
        
        return key, value


########################### CUDA Graph Cache ###########################

class GraphFlashSimpleCache(Cache):

    def __init__(self, model, max_budget=1024) -> None:

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            device=model.device
            dtype=torch.float16
            self.key_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
    
    def print_status(self):
        print("Max Budget:", self.max_budget)

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, storage_ids :torch.LongTensor, kv_cache=None, query_states=None, gamma_offset=0):

        input_length = len(storage_ids)

        assert input_length == new_k_cache.shape[-3], (input_length, new_k_cache.shape[-3])
        assert input_length == new_v_cache.shape[-3], (input_length, new_v_cache.shape[-3])
        # assert storage_ids[0].item() == gamma_offset, f"expected {gamma_offset}, got {storage_ids[0].item()}"
        
        self.key_cache[layer_idx].index_copy_(dim=-3, index=storage_ids, source=new_k_cache)
        self.value_cache[layer_idx].index_copy_(dim=-3, index=storage_ids, source=new_v_cache)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()


class GraphFlashStreamLLMCache(Cache):

    def __init__(self, model, max_budget=1024, prefill=2048, gamma=6, start_size=16) -> None:

        self.max_budget = max_budget
        self.gamma = gamma
        self.real_buget = self.max_budget + gamma

        self.prefill = prefill
        self.start_size = start_size
        self.recent_size = self.max_budget - self.start_size

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, 1, self.real_buget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_buget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
    
    def print_status(self):
        print("Max Budget:", self.max_budget, " | Real Budget:", self.real_buget, " | PreFill:", self.prefill, " | Start Size:", self.start_size, " | Recent Size:", self.recent_size)

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, storage_ids :torch.LongTensor, kv_cache=None, query_states=None, gamma_offset=0):

        input_length = len(storage_ids)

        assert input_length == new_k_cache.shape[-3], (input_length, new_k_cache.shape[-3])
        assert input_length == new_v_cache.shape[-3], (input_length, new_v_cache.shape[-3])
        
        # self.key_cache[layer_idx].index_copy_(dim=-3, index=storage_ids, source=new_k_cache)
        # self.value_cache[layer_idx].index_copy_(dim=-3, index=storage_ids, source=new_v_cache)

        self.key_cache[layer_idx][:, self.real_buget-self.gamma+gamma_offset] = new_k_cache
        self.value_cache[layer_idx][:, self.real_buget-self.gamma+gamma_offset] = new_v_cache

        return self.key_cache[layer_idx][:,:self.real_buget-self.gamma+gamma_offset+1], self.value_cache[layer_idx][:,:self.real_buget-self.gamma+gamma_offset+1]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def init_graph_cache(self, kv_cache):
        assert self.prefill == kv_cache.seq_len, f"expected prefill {self.prefill}, got {kv_cache.seq_len}"

        for layer in range(self.layers):
            self.key_cache[layer][:, :self.start_size] = kv_cache.key_cache[layer][:, :self.start_size]
            self.key_cache[layer][:, self.start_size:-self.gamma] = kv_cache.key_cache[layer][:, -self.recent_size + self.prefill:self.prefill]
            self.value_cache[layer][:, :self.start_size] = kv_cache.value_cache[layer][:, :self.start_size]
            self.value_cache[layer][:, self.start_size:-self.gamma] = kv_cache.value_cache[layer][:, -self.recent_size + self.prefill:self.prefill]

    def update_graph_cache(self, kv_cache):
        # !!! can be optimized (we can only replace one part of it!)
        self.key_cache[:,:, self.start_size:-self.gamma] = kv_cache.key_cache[:,:, -self.recent_size + kv_cache.seq_len:kv_cache.seq_len]
        self.value_cache[:,:, self.start_size:-self.gamma] = kv_cache.value_cache[:,:, -self.recent_size + kv_cache.seq_len:kv_cache.seq_len]

class GraphSimpleCache(Cache):

    def __init__(self, model, max_budget=1024) -> None:

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            if hasattr(model, 'gpt_neox'):
                device = model.gpt_neox.layers[i].attention.query_key_value.weight.device
                dtype = model.gpt_neox.layers[i].attention.query_key_value.weight.dtype
            else:
                # device = model.model.layers[i].self_attn.q_proj.weight.device
                # dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
                device=model.device
                dtype=torch.float16
            self.key_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.num_heads, self.max_budget, self.head_dim], dtype=dtype).to(device))
    
    def print_status(self):
        print("Max Budget:", self.max_budget)

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, storage_ids :torch.LongTensor, kv_cache=None, query_states=None):

        input_length = len(storage_ids)

        assert input_length == new_k_cache.shape[-2], (input_length, new_k_cache.shape[-2])
        assert input_length == new_v_cache.shape[-2], (input_length, new_v_cache.shape[-2])
        
        # print(storage_ids)
        self.key_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.value_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

class GraphFlashChunkCache(Cache):
    def __init__(self, model, prefill=1024, chunk_size=128, gamma=6, budget=0.1) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.chunk_k: List[torch.Tensor] = []
    
        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = int(budget * self.chunks)
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"

        self.max_budget = self.chunk_size * self.select_sets + gamma # max budget should be self.chunk_size * select sets + gamma

        self.hidden_size = model.config.hidden_size
        if hasattr(model.config, 'num_key_value_heads'):
            self.num_heads = model.config.num_key_value_heads
        else:
            self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        for i in range(self.layers):
            device=model.device
            dtype=torch.float16
            self.key_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))

            self.chunk_k.append(torch.zeros([1, self.chunks, self.num_heads, self.head_dim], dtype=dtype).to(device))
    
    def print_status(self):
        print("Max Budget:", self.max_budget)

    def update_chunk_k(self):
        for layer in range(self.layers):
            self.chunk_k[layer] = self.key_cache[layer][:, :, :self.prefill].view(1, self.num_heads, self.chunks, self.chunk_size, self.head_dim).mean(dim=-2)

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, storage_ids :torch.LongTensor, kv_cache=None, query_states=None):

        assert kv_cache is not None, "kv_cache should not be None"
        assert query_states is not None, "query_states should not be None"
        input_length = len(storage_ids)

        assert input_length == new_k_cache.shape[-2], (input_length, new_k_cache.shape[-2])
        assert input_length == new_v_cache.shape[-2], (input_length, new_v_cache.shape[-2])
        
        self.key_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.value_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # chunk_k: (bsz, chunks, 32, head_dim) --> (bsz, 32, head_dim, chunks)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), self.chunk_k[layer_idx].permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx = torch.topk(chunk_attn, k=self.select_sets, dim=-1).permute(0, 2, 1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache[layer_idx][:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        key_ = key_.permute(0, 1, 3, 2, 4)

    

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
            self.chunk_k[i].zero_()
