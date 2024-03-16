from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from time import sleep
import math
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union
from .TP_layers import DistributedLlamaLayer, DistributedLlamaLayerBuffer, DistributedOffloadingConfig
from .tensor_op import RMSNorm, TP_MLP, TP_Attention
import torch.distributed as dist
from .configuration_llama import LlamaConfig
from .batch_cache import DistributedBatchSimpleCache

def distributed_init():
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    return local_rank, world_size

class DistributedLlama:
    def __init__(self, 
        model_name_or_path: str, 
        dtype = torch.float16,
        kv_offload = False,
        on_chip_layers = 24,
        local_rank = 0,
        world_size = 1) -> None:
        
        self.device  = torch.device("cuda", local_rank)
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_size = world_size
        self.kv_offload = kv_offload
        self.on_chip_layers = on_chip_layers
        model_config: LlamaConfig = LlamaConfig.from_pretrained(model_name_or_path)
        self.config = DistributedOffloadingConfig(model_config, local_rank, world_size)

        self.kv_cache = DistributedBatchSimpleCache(self.config, max_budget=1024, bsz=2, device=self.device)
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = model_config.max_position_embeddings
        self.rope_theta =  model_config.rope_theta

        self.local_num_heads = self.num_heads // world_size
        self.local_num_key_value_heads = self.num_key_value_heads // world_size
    
    def init_parameters(self, hf_model: LlamaForCausalLM):

        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.layers :list[DistributedLlamaLayer] = []
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = DistributedLlamaLayer(idx, self.config)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
        
        self.num_layers = len(self.layers)
        self.buffer = DistributedLlamaLayerBuffer(self.config)
        self.buffer.init_space(self.layers[0])
        for id in range(self.on_chip_layers):
            self.layers[id].to_gpu()

    def layer_compute(self, 
            buffer: Union[DistributedLlamaLayerBuffer, DistributedLlamaLayer],
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor):

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.layers[layer_idx].input_layernorm_variance_epsilon,
            layernorm_weight=self.layers[layer_idx].input_layernorm_weight
        )
        
        hidden_states = TP_Attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=layer_idx,
            wq=buffer.wq,
            wk=buffer.wk,
            wv=buffer.wv,
            wo=buffer.wo,
            sin_cache=self.layers[layer_idx].sin_cache,
            cos_cache=self.layers[layer_idx].cos_cache,
            kv_buffer=self.kv_buffer if self.kv_offload else self.kv_cache,
            hidden_size=self.hidden_size,
            local_num_heads=self.local_num_heads,
            local_num_key_value_heads=self.local_num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim
            
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.layers[layer_idx].post_attention_layernorm_variance_epsilon,
            layernorm_weight=self.layers[layer_idx].post_attention_layernorm_weight
        )

        hidden_states = TP_MLP(
            hidden_states=hidden_states,
            up_proj=buffer.up_proj,
            down_proj=buffer.down_proj,
            gate_proj=buffer.gate_proj
        )

        
        hidden_states = residual + hidden_states
        return hidden_states

    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor):
        
        kv_len = self.kv_cache.kv_offset
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        if self.kv_offload:
            for idx in range(self.num_layers):
                if idx >= self.on_chip_layers:
                    self.buffer.sync_copy(self.layers[idx])
                    self.kv_buffer.copy_kv(
                        self.kv_cache.k_cache[idx],
                        self.kv_cache.v_cache[idx],
                        kv_len
                    )
                    hidden_states = self.layer_compute(self.buffer, idx, hidden_states, position_ids, attention_mask)
                    self.kv_cache.copy_back_from_buffer(self.kv_buffer, idx)
                else:
                    self.kv_buffer.copy_kv(
                        self.kv_cache.k_cache[idx],
                        self.kv_cache.v_cache[idx],
                        kv_len
                    )
                    hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask)
                    self.kv_cache.copy_back_from_buffer(self.kv_buffer, idx)

        else:
            for idx in range(self.num_layers):
                if idx < self.on_chip_layers:
                    hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask)
                else:
                    self.buffer.sync_copy(self.layers[idx])
                    hidden_states = self.layer_compute(self.buffer, idx, hidden_states, position_ids, attention_mask)

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.norm_variance_epsilon,
            layernorm_weight=self.norm_weight
        )
        
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits