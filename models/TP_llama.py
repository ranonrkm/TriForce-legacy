from tkinter import NO
from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from time import sleep
import math
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union
from .TP_layers import DistributedLlamaLayer, DistributedLlamaLayerBuffer, DistributedOffloadingConfig
from .tensor_op import RMSNorm, TP_MLP, TP_Attention, TP_Attention_Retrieval
import torch.distributed as dist
from .configuration_llama import LlamaConfig
from .batch_cache import DistributedBatchSimpleCache, DistributedBatchRetrievalCache
from utils.sampling import norm_logits

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
        on_chip_layers = 32,
        local_rank = 0,
        world_size = 1,
        prefill = 32768,
        bsz = 2,
        gen_len = 256,
        retrieval_budget = 4096,
        retrieval_chunk_size = 8,
        gamma = 6,
        temperature = 0.6,
        top_p = 0.9,
        flash_attn=True) -> None:
        
        self.device  = torch.device("cuda", local_rank)
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_size = world_size
        self.kv_offload = kv_offload
        self.on_chip_layers = on_chip_layers
        self.flash_attn = flash_attn
        model_config: LlamaConfig = LlamaConfig.from_pretrained(model_name_or_path)
        self.config = DistributedOffloadingConfig(model_config, local_rank, world_size)
        self.vocab_size = model_config.vocab_size

        self.temperature = temperature
        self.top_p = top_p
        
        if kv_offload:
            self.kv_cache = DistributedBatchSimpleCache(self.config, max_budget=prefill+gen_len, bsz=bsz, device='cpu')
            raise NotImplementedError("KV offload is not supported")
            # self.kv_buffer = KVCacheBuffer(self.config, device=self.device, dtype=dtype)
        else:
            self.kv_cache = DistributedBatchSimpleCache(self.config, max_budget=prefill+3*gen_len, bsz=bsz, device=self.device)
            self.retrieval_cache = DistributedBatchRetrievalCache(self.config, max_budget=retrieval_budget, bsz=bsz, device=self.device, prefill=prefill, chunk_size=retrieval_chunk_size, gamma=gamma)

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
        # self.buffer = DistributedLlamaLayerBuffer(self.config)
        # self.buffer.init_space(self.layers[0])
        for id in range(self.on_chip_layers):
            self.layers[id].to_gpu(device=self.device)

    def layer_compute(self, 
            buffer: Union[DistributedLlamaLayerBuffer, DistributedLlamaLayer],
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor=None, 
            attention_mask: torch.FloatTensor=None,
            retrieval_cache=None):

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
            head_dim=self.head_dim,
            flash_attn=self.flash_attn,
            retrieval_cache=retrieval_cache
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
    
    def reset(self):
        self.kv_cache.reset()
        self.retrieval_cache.reset()

    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor=None,
            attention_mask: torch.FloatTensor=None,
            retrieval_cache=None):
        
        # kv_len = self.kv_cache.kv_offset
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        if position_ids is None:
            batch_size, seq_length = input_ids.shape[:2]
            range_tensor = torch.arange(seq_length, dtype=torch.long, device=self.device)
            # print(kv_cache.seq_len, range_tensor)
            position_ids = self.kv_cache.seq_len[:, None] + range_tensor

        if self.kv_offload:
            raise NotImplementedError("KV offload is not supported")
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
                    # print(f"Rank {self.local_rank} is computing layer {idx} with {hidden_states.device} using {self.layers[idx].wq.device}")
                    hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, retrieval_cache)
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

    def prefill(self, input_ids: torch.LongTensor):
        iter_prefill = math.ceil(input_ids.shape[1] / 64)
        for i in range(iter_prefill):
            logits = self.inference(input_ids=input_ids[:, i*64:(i+1)*64])
        return logits
    
    def build_retrieval_cache(self, input_ids: torch.LongTensor):
        assert input_ids.shape[-1] == 1
        logits = self.inference(input_ids=input_ids, retrieval_cache=self.retrieval_cache)
        return logits
    

    def layer_speculation(self, 
            buffer: Union[DistributedLlamaLayerBuffer, DistributedLlamaLayer],
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor=None, 
            attention_mask: torch.FloatTensor=None,
            retrieval_cache=None,
            gamma_offset:int=-1):

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.layers[layer_idx].input_layernorm_variance_epsilon,
            layernorm_weight=self.layers[layer_idx].input_layernorm_weight
        )
        
        hidden_states = TP_Attention_Retrieval(
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
            hidden_size=self.hidden_size,
            local_num_heads=self.local_num_heads,
            local_num_key_value_heads=self.local_num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            flash_attn=self.flash_attn,
            retrieval_cache=retrieval_cache,
            gamma_offset=gamma_offset
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
    
    def retrieval_inference(self, input_ids: torch.LongTensor, gamma_offset: int, position_ids: torch.LongTensor):
        # kv_len = self.kv_cache.kv_offset
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            # print(f"Rank {self.local_rank} is computing layer {idx} with {hidden_states.device} using {self.layers[idx].wq.device}")
            hidden_states = self.layer_speculation(self.layers[idx], idx, hidden_states, position_ids, attention_mask=None, retrieval_cache=self.retrieval_cache, gamma_offset=gamma_offset)

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.norm_variance_epsilon,
            layernorm_weight=self.norm_weight
        )

        logits = F.linear(hidden_states, self.lm_head).float()
        return norm_logits(logits[:,-1,:], temperature=self.temperature, top_k=-1, top_p=self.top_p)