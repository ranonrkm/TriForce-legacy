import select
from typing import Any, Dict, List, Optional, Tuple

import torch
import copy

import time

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length


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
                device = model.model.layers[i].self_attn.q_proj.weight.device
                dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
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
    
class FlashSimpleCache(Cache):
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
                device = model.model.layers[i].self_attn.q_proj.weight.device
                dtype = model.model.layers[i].self_attn.q_proj.weight.dtype
            # print(device, dtype)
            self.key_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))
            self.value_cache.append(torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(device))

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
    def __init__(self, model, max_budget=1024, start_size=4, recent_size=16, skip_start_layers=-1) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_len = 0
        self.start_size = start_size
        self.recent_size = recent_size
        self.max_budget = max_budget
        self.skip_start_layers = skip_start_layers

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

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget, "| Start Size:", self.start_size, "| Recent Size:", self.recent_size)

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
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the cache

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + key_states.shape[-2]] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + value_states.shape[-2]] = value_states

        key = self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]
        value = self.value_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-2]

        return key, value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states"""
        return self.max_budget

    def speculation_update(self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Update the cache
        assert key_states.shape[-2] == 1

        if self.seq_len <= self.start_size + self.recent_size:
            return self.update(key_states, value_states, layer_idx, cache_kwargs)

        self.key_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len : self.seq_len + 1] = value_states

        key = torch.cat([
            self.key_cache[layer_idx][:, :, :self.start_size], 
            self.key_cache[layer_idx][:, :, -self.recent_size + self.seq_len:self.seq_len + 1]
        ], dim=-2)
        value = torch.cat([
            self.value_cache[layer_idx][:, :, :self.start_size], 
            self.value_cache[layer_idx][:, :, -self.recent_size + self.seq_len:self.seq_len + 1]
        ], dim=-2)

        if layer_idx == self.layers-1:
            self.seq_len += 1

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


class EfficientH2OCache(Cache):
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

        # print(attn_weights.shape, self.hh_score[layer_idx][:, :, :attn_weights.shape[-1]].shape)
        self.hh_score[layer_idx][:, :, :attn_weights.shape[-1]] += attn_weights


    def update_heavy_cache(self):
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
        import math
        attn_weights = torch.matmul(query_states, self.key_cache[layer_idx][:, :, :self.seq_len + value_states.shape[-2]].transpose(2, 3)) / math.sqrt(self.head_dim) # (bsz, 32, 1, kv_seq_len)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        assert attn_weights.shape[2] == 1
        attn_weights = attn_weights.squeeze(2) # (bsz, 32, kv_seq_len)
        _, topk_idx = torch.topk(attn_weights, k=self.topk_size, dim=-1)
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