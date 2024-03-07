import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from typing import List, Optional, Tuple, Union
import gc
import math
from tqdm import tqdm

class InferenceEngine:
    def __init__(self, model, cache, graph_cache) -> None:
        self.model = model
        self.model.eval()
        self.model_config = self.model.config
        self.kv_cache = cache
        self.graph_cache = graph_cache
    
    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        if input_ids.shape[-1] > 1: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 100)
            for i in tqdm(range(iter_prefill)):
                logits = self.model(
                    input_ids=input_ids[:, i*100:(i+1)*100],
                    kv_cache=self.kv_cache,
                    graph_cache=None,
                ).logits
        else: # init graph cache, input_ids.shape[-1] == 1
            logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache).logits
        return logits
    
    @torch.inference_mode()
    def verify(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=None, position_ids=position_ids, attention_mask=attention_mask).logits
        return logits
    
    
    @torch.inference_mode()
    def draft(self, input_ids: torch.LongTensor, storage_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        logits = self.model(input_ids=input_ids, kv_cache=None, graph_cache=self.graph_cache, storage_ids=storage_ids, position_ids=position_ids, attention_mask=attention_mask, spec=True).logits
        return logits

    def clear_kv(self):
        self.graph_cache.reset()
        self.kv_cache.reset()


def capture_graph(engine :InferenceEngine, decoding_seqlen :int =1, mempool=None, n_warmups :int=3):
    device = engine.model.device
    dtype=torch.float16

    static_input_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, engine.max_length), 0, dtype=dtype, device=device)
    static_attn_mask = static_attn_mask[None, None, :, :]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.draft(
                    input_ids=static_input_ids, 
                    storage_ids=static_storage_ids, 
                    position_ids=static_position_ids, 
                    attention_mask=static_attn_mask
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.draft(
                input_ids=static_input_ids, 
                storage_ids=static_storage_ids, 
                position_ids=static_position_ids, 
                attention_mask=static_attn_mask
                )
    def run(input_ids, storage_ids, position_ids, attn_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attn_mask)
        graph.replay()
        return static_logits.clone()
    
    return run

class GraphInferenceEngine:
    def __init__(self, model, cache, graph_cache) -> None:

        self.engine = InferenceEngine(model, cache, graph_cache)
        self.callables = {}
        self.mempool = None

    @torch.inference_mode()
    def initialize_cuda_graph(self, decoding_seqlens :List[int]):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()

        for decoding_seqlen in decoding_seqlens:
            if decoding_seqlen not in self.callables:
                self.callables[decoding_seqlen] = capture_graph(
                    engine=self.engine,
                    decoding_seqlen=decoding_seqlen,
                    mempool=self.mempool,
                    n_warmups=3
                )
        self.engine.clear_kv()

    @torch.inference_mode()
    def graph_inference(self, input_ids: torch.LongTensor, storage_ids: torch.LongTensor, position_ids: torch.LongTensor, attn_mask):
        dec_length = input_ids.shape[1]
        logits = self.callables[dec_length](input_ids, storage_ids, position_ids, attn_mask)
        return logits
    
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, storage_ids: torch.LongTensor, position_ids: torch.LongTensor, attn_mask):
        logits = self.engine.draft(input_ids, storage_ids, position_ids, attn_mask)
        return logits

    def clear_kv(self):
        self.engine.clear_kv()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        return self.engine.prefill(input_ids=input_ids)

    @torch.inference_mode()
    def verify(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        return self.engine.verify(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    def update_graph_cache(self):
        self.engine.graph_cache.update_graph_cache(kv_cache=self.engine.kv_cache)

    def kv_stats(self):
        self.engine.kv_cache.print_status()
        self.engine.graph_cache.print_status()



################## UTILS ##################

from torch.nn.functional import softmax
def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1))
    return residual

def sampling_without_replacement(
        sampling_logits: torch.Tensor, 
        rand: torch.Tensor,  
        num_samples: int,
        temperature :float):

        sampling_q = softmax(sampling_logits / temperature, dim=-1)
        position = (rand.log()/sampling_q).topk(k=num_samples).indices.flatten()
        return position

def sampling_with_replacement(
        sampling_logits: torch.Tensor,   
        num_samples: int,
        temperature :float):

        #sampling_q = softmax(sampling_logits / temperature, dim=-1)
        sampling_q = softmax(sampling_logits / temperature, dim=-1)    
        position = sampling_q.multinomial(num_samples=num_samples, replacement=False).flatten()
        return position
def sampling_argmax(
        sampling_logits: torch.Tensor, 
        num_samples: int):
        return sampling_logits.topk(k=num_samples).indices.flatten()

def expand_kv(kv_cache, k):
    kv_shape = kv_cache[0][0].shape
    new_kv_cache = ()
    for kv in kv_cache:
        new_kv_cache = new_kv_cache + ([kv[0].expand(k, kv_shape[1], kv_shape[2], kv_shape[3]), 
                kv[1].expand(k, kv_shape[1], kv_shape[2], kv_shape[3])],)
    return new_kv_cache

def cat_kv(old_kv, delta_kv, cut_len :int):
    new_kv_cache = ()
    for i in range(len(old_kv)):
          k = torch.cat([old_kv[i][0], delta_kv[i][0][..., -cut_len:, :]], dim=-2)
          v = torch.cat([old_kv[i][1], delta_kv[i][1][..., -cut_len:, :]], dim=-2)
          new_kv_cache += ([k,v],)
    return new_kv_cache
    
    
def make_tree_attention_mask(
        prefix_len :int,
        gen_len :int,
        ancestors :list[list[int]],
        device ="cpu",
        dtype = torch.float32
    ) -> torch.FloatTensor:
    tree_mask = torch.full((gen_len, gen_len + prefix_len), torch.finfo(dtype).min, dtype=dtype).to(device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :]


def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def select_kv(kv_cache: tuple[list[torch.FloatTensor]], indices: list[int]):
    new_kv_cache = ()
    for k,v in kv_cache:
            k = k[..., indices, :]
            v = v[..., indices, :]
            new_kv_cache += ([k,v],)
    return new_kv_cache


def cuda_graph_for_residual(device="cuda:0", dtype=torch.float16, dim=32000, n_warmups=3, mempool=None):
    static_p = torch.full((dim,), 1, dtype=dtype, device=device)
    static_q = torch.full((dim,), 0, dtype=dtype, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_residual = get_residual(
                    static_p,
                    static_q
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         static_residual = get_residual(
                    static_p,
                    static_q
                    )
    def run(p, q):
        static_p.copy_(p)
        static_q.copy_(q)
        graph.replay()
        return static_residual.clone()
    
    return run

def cuda_graph_for_sampling_without_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    static_rand = torch.empty((idx_len, dim), dtype=dtype, device=device).uniform_()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
    def run(draft_logits, rand_vector):
        static_sampling_logits.copy_(draft_logits)
        static_rand.copy_(rand_vector)
        graph.replay()
        return static_position.clone()
    
    return run

def cuda_graph_for_sampling_argmax(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run


def cuda_graph_for_sampling_with_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run