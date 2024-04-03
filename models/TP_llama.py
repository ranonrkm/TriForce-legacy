from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from time import sleep
import math
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union
import gc
from tqdm import tqdm

from .TP_layers import DistributedLlamaLayer, DistributedLlamaLayerBuffer, DistributedOffloadingConfig
from .tensor_op import RMSNorm, TP_MLP, TP_Attention, TP_Attention_Retrieval, TP_Attention_Tree_Retrieval, TP_Attention_ssl
import torch.distributed as dist
from .configuration_llama import LlamaConfig
from .batch_cache import DistributedBatchSimpleCache, DistributedBatchRetrievalCache, DistributedBatchKVCacheBuffer
from .cache_utils import DistributedKVCacheBuffer, DistributedSimpleCache, DistributedRetrievalCache
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
        bsz = 1,
        gen_len = 256,
        retrieval_budget = 4096,
        retrieval_chunk_size = 8,
        gamma = 6,
        temperature = 0.6,
        top_p = 0.9,
        tree_size=128,
        ssl=0,
        flash_attn=True) -> None:
        
        self.device  = torch.device("cuda", local_rank)
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_size = world_size
        self.kv_offload = kv_offload
        self.on_chip_layers = on_chip_layers
        self.ssl = ssl
        self.flash_attn = flash_attn
        model_config: LlamaConfig = LlamaConfig.from_pretrained(model_name_or_path)
        self.config = DistributedOffloadingConfig(model_config, local_rank, world_size)
        self.vocab_size = model_config.vocab_size
        self.prefill_len = prefill
        self.retrieval_budget = retrieval_budget
        self.temperature = temperature
        self.top_p = top_p
        self.gamma = gamma
        self.bsz = bsz
        self.load_stream = torch.cuda.Stream(device=self.device)
        
        if kv_offload:
            assert bsz == 1
            # self.kv_cache = DistributedBatchSimpleCache(self.config, max_budget=prefill+3*gen_len, bsz=bsz, device='cpu')
            # self.kv_buffer = DistributedBatchKVCacheBuffer(self.config, max_budget=prefill+3*gen_len, bsz=bsz, device=self.device)
            self.kv_cache =  DistributedSimpleCache(self.config, max_budget=prefill+gen_len+tree_size, device=self.device, on_chip_layers=on_chip_layers, ssl=ssl)
            self.kv_buffer = [DistributedKVCacheBuffer(self.config, max_budget=prefill+gen_len+tree_size, device=self.device) for _ in range(2)]
            self.retrieval_cache = DistributedRetrievalCache(self.config, max_budget=retrieval_budget, device=self.device, prefill=prefill, chunk_size=retrieval_chunk_size, tree_size=tree_size)
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
            if idx == 0:
                self.sin_cache = hf_layer.self_attn.rotary_emb.sin_cached.to(self.device)
                self.cos_cache = hf_layer.self_attn.rotary_emb.cos_cached.to(self.device)
            layer = DistributedLlamaLayer(idx, self.config)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)

        self.num_layers = len(self.layers)
        for id in range(self.num_layers):
            self.layers[id].to_gpu(device=self.device)

    @torch.inference_mode()
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
            sin_cache=self.sin_cache,
            cos_cache=self.cos_cache,
            kv_buffer=self.kv_buffer[(layer_idx) % 2] if (layer_idx >= self.on_chip_layers) else self.kv_cache,
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

    @torch.inference_mode()
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
            if batch_size > 1:
                position_ids = self.kv_cache.seq_len[:, None] + range_tensor
            else:
                position_ids = self.kv_cache.seq_len + range_tensor
                position_ids = position_ids.unsqueeze(0)

        if self.kv_offload:
            self.kv_buffer[(self.on_chip_layers) % 2].copy_kv(self.kv_cache, self.on_chip_layers)
            # torch.cuda.synchronize()
            for idx in range(self.num_layers):
                if idx >= self.on_chip_layers:
                    torch.cuda.synchronize() # !!! MUST
                    with torch.cuda.stream(self.load_stream):
                        hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, retrieval_cache)
                        self.kv_cache.copy_back_from_buffer(self.kv_buffer[(idx) % 2], idx)
                    # torch.cuda.current_stream().wait_stream(self.load_stream)
                    # self.load_stream.wait_stream(torch.cuda.current_stream())
                    # self.load_stream.synchronize()
                    if idx != self.num_layers - 1:
                        self.kv_buffer[(idx + 1) % 2].copy_kv(self.kv_cache, idx + 1)
                    torch.cuda.synchronize()
                # if idx >= self.on_chip_layers:
                #     self.kv_buffer[(idx) % 2].copy_kv(self.kv_cache, idx)
                #     hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, retrieval_cache)
                #     self.kv_cache.copy_back_from_buffer(self.kv_buffer[(idx) % 2], idx)
                else:
                    hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, retrieval_cache)

        else:
            for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, retrieval_cache)

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.norm_variance_epsilon,
            layernorm_weight=self.norm_weight
        )

        logits = F.linear(hidden_states, self.lm_head)#.float()
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        iter_prefill = math.ceil(input_ids.shape[1] / 128)
        for i in tqdm(range(iter_prefill)):
            logits = self.inference(input_ids=input_ids[:, i*128:(i+1)*128])
        return logits

    @torch.inference_mode()
    def build_retrieval_cache(self, input_ids: torch.LongTensor):
        assert input_ids.shape[-1] == 1
        logits = self.inference(input_ids=input_ids, retrieval_cache=self.retrieval_cache)
        return logits

    @torch.inference_mode()
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
            sin_cache=self.sin_cache,
            cos_cache=self.cos_cache,
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

    @torch.inference_mode()
    def layer_tree_speculation(self, 
            buffer: Union[DistributedLlamaLayerBuffer, DistributedLlamaLayer],
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor=None, 
            attention_mask: torch.FloatTensor=None,
            retrieval_cache=None,
            storage_ids=None):

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.layers[layer_idx].input_layernorm_variance_epsilon,
            layernorm_weight=self.layers[layer_idx].input_layernorm_weight
        )
        
        hidden_states = TP_Attention_Tree_Retrieval(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=layer_idx,
            wq=buffer.wq,
            wk=buffer.wk,
            wv=buffer.wv,
            wo=buffer.wo,
            sin_cache=self.sin_cache,
            cos_cache=self.cos_cache,
            hidden_size=self.hidden_size,
            local_num_heads=self.local_num_heads,
            local_num_key_value_heads=self.local_num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            flash_attn='sdpa',
            retrieval_cache=retrieval_cache,
            storage_ids=storage_ids
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

    @torch.inference_mode()
    def layer_compute_ssl(self, 
            buffer: Union[DistributedLlamaLayerBuffer, DistributedLlamaLayer],
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor=None, 
            attention_mask: torch.FloatTensor=None):

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.layers[layer_idx].input_layernorm_variance_epsilon,
            layernorm_weight=self.layers[layer_idx].input_layernorm_weight
        )

        hidden_states = TP_Attention_ssl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=layer_idx,
            wq=buffer.wq,
            wk=buffer.wk,
            wv=buffer.wv,
            wo=buffer.wo,
            sin_cache=self.sin_cache,
            cos_cache=self.cos_cache,
            kv_buffer=self.kv_buffer[(layer_idx) % 2] if (layer_idx >= self.on_chip_layers) else self.kv_cache,
            hidden_size=self.hidden_size,
            local_num_heads=self.local_num_heads,
            local_num_key_value_heads=self.local_num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            flash_attn=self.flash_attn,
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


    @torch.inference_mode()
    def retrieval_tree_inference(self, input_ids: torch.LongTensor, storage_ids, position_ids, attention_mask):
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        if self.ssl > 0:
            # print(torch.zeros(1, 1, input_ids.shape[-1], self.kv_cache.seq_len, device=self.device).shape, attention_mask.shape, attention_mask[:,:,:self.kv_cache.ssl_cur + input_ids.shape[-1]].shape)
            ssl_mask = torch.cat([torch.zeros(1, 1, input_ids.shape[-1], self.kv_cache.seq_len, device=self.device), attention_mask[:,:,:,:self.kv_cache.ssl_cur + input_ids.shape[-1]]], dim=-1)

        for idx in range(self.num_layers):
            if idx >= self.ssl:
                hidden_states = self.layer_tree_speculation(self.layers[idx], idx, hidden_states, position_ids, attention_mask=attention_mask, storage_ids=storage_ids, retrieval_cache=self.retrieval_cache)
            else:
                hidden_states = self.layer_compute_ssl(self.layers[idx], idx, hidden_states, position_ids, attention_mask=ssl_mask)

        hidden_states = RMSNorm(
            hidden_states=hidden_states,
            layernorm_variance_epsilon=self.norm_variance_epsilon,
            layernorm_weight=self.norm_weight
        )

        logits = F.linear(hidden_states, self.lm_head)#.float()
        return logits


    # @torch.inference_mode()
    # def retrieval_inference(self, input_ids: torch.LongTensor, gamma_offset: int, position_ids: torch.LongTensor):
    #     hidden_states = F.embedding(input_ids, self.embed_tokens)

    #     for idx in range(self.num_layers):
    #         hidden_states = self.layer_speculation(self.layers[idx], idx, hidden_states, position_ids, attention_mask=None, retrieval_cache=self.retrieval_cache, gamma_offset=gamma_offset)

    #     hidden_states = RMSNorm(
    #         hidden_states=hidden_states,
    #         layernorm_variance_epsilon=self.norm_variance_epsilon,
    #         layernorm_weight=self.norm_weight
    #     )

    #     logits = F.linear(hidden_states, self.lm_head)#.float()
    #     return norm_logits(logits[:,-1,:], temperature=self.temperature, top_k=-1, top_p=self.top_p)

    @torch.inference_mode()
    def capture_graph_retrieval_inference(self, gamma_offset: int, mempool, n_warmups: int):
        
        static_input_ids = torch.full((self.bsz, 1), 0, dtype=torch.long, device=self.device)
        static_position_ids = torch.full((self.bsz, 1), 1024, dtype=torch.long, device=self.device)
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_logits = self.retrieval_inference(input_ids=static_input_ids, gamma_offset=gamma_offset, position_ids=static_position_ids)
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)
        if self.local_rank == 0:
            print(f"[retrieval run] capturing graph for spec len {gamma_offset} (temp={self.temperature}, top_p={self.top_p})...")
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=mempool):
            static_logits = self.retrieval_inference(input_ids=static_input_ids, gamma_offset=gamma_offset, position_ids=static_position_ids)
        dist.barrier()
        def run(input_ids, position_ids):
            static_input_ids.copy_(input_ids)
            static_position_ids.copy_(position_ids)
            graph.replay()
            return static_logits.clone()
        
        return run

    @torch.inference_mode()
    def initialize_cuda_graph(self):
        self.callables = {}
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()

        for gamma_offset in range(self.gamma):
            self.callables[gamma_offset] = self.capture_graph_retrieval_inference(
                                                gamma_offset=gamma_offset,
                                                mempool=self.mempool,
                                                n_warmups=24,
                                            )

        self.reset()

    @torch.inference_mode()
    def retrieval_graph_inference(self, input_ids: torch.LongTensor, gamma_offset: int, position_ids: torch.LongTensor):
        return self.callables[gamma_offset](input_ids, position_ids)