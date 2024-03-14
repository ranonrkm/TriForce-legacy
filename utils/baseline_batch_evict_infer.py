import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from typing import List, Optional, Tuple, Union
import gc
import math
from tqdm import tqdm

from .sampling import norm_logits

class InferenceEngine:
    def __init__(self, model, cache, draft, draft_cache, bsz=2) -> None:

        ###### 7B ######
        self.model = model
        self.model.eval()
        self.kv_cache = cache
        self.bsz = bsz

        ###### 68 MB ######
        self.draft = draft
        self.draft.eval()
        self.draft_cache = draft_cache

    @torch.inference_mode()
    def draft_run(self, input_ids: torch.LongTensor, gamma_offset: int=0, probs=False, temperature=0.6, top_p=0.9):
        if input_ids.shape[-1] >= 8: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 8)
            for i in range(iter_prefill):
                self.draft_cache.evict_prefill(8)
                logits = self.draft(
                    input_ids=input_ids[:, i*8:(i+1)*8],
                    kv_cache=self.draft_cache,
                    graph_cache=None,
                    gamma_offset=-1
                ).logits
        else: # decoding
            logits = self.draft(input_ids=input_ids, kv_cache=self.draft_cache, graph_cache=self.draft_cache, gamma_offset=gamma_offset).logits

        if probs:
            return norm_logits(logits[:,-1,:], temperature=temperature, top_k=-1, top_p=top_p)
        return logits

    @torch.inference_mode()
    def model_run(self, input_ids: torch.LongTensor):
        if input_ids.shape[-1] > 64: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 64)
            for i in range(iter_prefill):
                logits = self.model(
                    input_ids=input_ids[:, i*64:(i+1)*64],
                    kv_cache=self.kv_cache,
                    graph_cache=None,
                ).logits
        else: # verification
            logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache).logits
        return logits

    def clear_kv(self):
        self.kv_cache.reset()
        self.draft_cache.reset()

def draft_run_capture_graph(engine :InferenceEngine, gamma_offset :int =0, mempool=None, n_warmups :int=3, probs=False, temperature=0.6, top_p=0.9):
    device = engine.draft.device
    bsz = engine.bsz
    # draft run is incremental decoding
    static_input_ids = torch.full((bsz, 1), 0, dtype=torch.long, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.draft_run(input_ids=static_input_ids, gamma_offset=gamma_offset, probs=probs, temperature=temperature, top_p=top_p)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print(f"[draft run] capturing graph for {gamma_offset} (probs={probs}, temp={temperature}, top_p={top_p})...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.draft_run(input_ids=static_input_ids, gamma_offset=gamma_offset, probs=probs, temperature=temperature, top_p=top_p)
    def run(input_ids):
        static_input_ids.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    return run


class GraphInferenceEngine:
    def __init__(self, model, cache, draft, draft_cache, bsz) -> None:

        self.engine = InferenceEngine(model, cache, draft, draft_cache, bsz)
        self.callables = {}
        self.mempool = None

    @torch.inference_mode()
    def initialize_cuda_graph(self, gamma=6, probs=False, temperature=0.6, top_p=0.9):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        


        for gamma_offset in range(gamma+1):
            self.callables[gamma_offset] = draft_run_capture_graph(
                                                engine=self.engine,
                                                gamma_offset=gamma_offset,
                                                mempool=self.mempool,
                                                n_warmups=3,
                                                probs=probs,
                                                temperature=temperature,
                                                top_p=top_p
                                            )

        self.engine.clear_kv()

    def clear_kv(self):
        self.engine.clear_kv()

    @torch.inference_mode()
    def graph_draft_inference(self, input_ids: torch.LongTensor, gamma_offset: int=0):
        # draft run
        return self.callables[gamma_offset](input_ids)
    
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor):
        # model run
        return self.engine.model_run(input_ids=input_ids)
    
    @torch.inference_mode()
    def graph_draft_prefill(self, input_ids: torch.LongTensor):
        # draft run
        logits = self.engine.draft_run(input_ids=input_ids)
        return logits



