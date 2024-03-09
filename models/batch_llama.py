from typing import List, Optional, Tuple, Union
from numpy import zeros_like

import torch
import torch.nn.functional as F
from torch import nn, ones_like
from flash_attn import flash_attn_with_kvcache
from transformers.activations import ACT2FN
from models.cache_utils import Cache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    #_prepare_4d_causal_attention_mask_for_sdpa,
)

from .modeling_llama_cache import LlamaYaRNRotaryEmbedding, LlamaRotaryEmbedding
import math
from transformers.models.llama.modeling_llama import(
    LlamaRMSNorm,
    LlamaConfig,
    PreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
    ACT2FN
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_llama import LlamaConfig
from models.cache_utils import Cache, TREEChunkTopKCache, TREESimpleCache


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, flash=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "yarn":
                original_max_position_embeddings = self.config.rope_scaling["original_max_position_embeddings"]
                self.rotary_emb = LlamaYaRNRotaryEmbedding(
                    self.head_dim, base=10000, scaling_factor=scaling_factor,
                    max_position_embeddings=self.max_position_embeddings,
                    original_max_position_embeddings=original_max_position_embeddings,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        spec=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states)
        # if spec:
        #     print(query_states.shape, key_states.shape, value_states.shape, position_ids, storage_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if spec: # spec decoding: graph cache
            key_states, value_states = graph_cache.update(new_k_cache=key_states, new_v_cache=value_states, layer_idx=self.layer_idx, storage_ids=storage_ids)
            # print(query_states.shape, key_states.shape, value_states.shape, position_ids)
        else:
            # update kv cache first
            key_states, value_states = kv_cache.update(key_states, value_states, layer_idx=self.layer_idx)
            # init graph cache (last prefill)
            if query_states.shape[2] == 1 and graph_cache is not None:
                graph_cache.init_graph_cache(kv_cache, query_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # print(query_states.shape, key_states.shape, value_states.shape, position_ids, storage_ids)
        
        if attention_mask is None:
            assert spec == False, "Attention mask is None only for the last prefill"
            assert query_states.shape[2] == 1, "Attention mask is None only for the last prefill"
        
            attn_weights = torch.matmul(query_states, key_states.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        else:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                attn_output = F.scaled_dot_product_attention(query_states,key_states,value_states, attn_mask=attention_mask.half())

            # attn_output = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), softmax_scale=1/torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float16)), causal=True).transpose(1, 2)

            # attn_weights = torch.matmul(query_states, key_states.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            # attn_weights = attn_weights + attention_mask
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_output = torch.matmul(attn_weights, value_states)


        # attn_weights = torch.matmul(query_states, key_states.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # if attention_mask is not None:
        #     attn_weights = attn_weights + attention_mask
        # else:
        #     assert spec == False, "Attention mask is None only for the last prefill"
        #     assert query_states.shape[2] == 1, "Attention mask is None only for the last prefill"
        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = (
            LlamaAttention(config=config, layer_idx=layer_idx)
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,

        spec=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            spec=spec,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,

        spec=False,
    ):
        batch_size, seq_length = input_ids.shape[:2]
        if position_ids is None:
            kv_cache_length = kv_cache.seq_len
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(kv_cache_length, seq_length + kv_cache_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if attention_mask is None and spec==False:
            # casual mask for prefilling
            past_key_values_length = kv_cache.seq_len
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                graph_cache=graph_cache,
                storage_ids=storage_ids,
                spec=spec,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        spec=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            spec=spec,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        return CausalLMOutputWithPast(
            logits=logits,
        )