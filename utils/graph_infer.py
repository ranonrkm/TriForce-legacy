import os
import sys
from turtle import position
from py import log
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
    def model_run(self, input_ids: torch.LongTensor, storage_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None):
        if storage_ids is None:
            if input_ids.shape[-1] > 1024: # prefill
                iter_prefill = math.ceil(input_ids.shape[1] / 100)
                for i in tqdm(range(iter_prefill)):
                    logits = self.model(
                        input_ids=input_ids[:, i*100:(i+1)*100],
                        kv_cache=self.kv_cache,
                        graph_cache=None,
                    ).logits
            else:
                logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=None).logits
        else: # graph
            logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache, storage_ids=storage_ids, position_ids=position_ids).logits
        return logits
    
    def clear_kv(self):
        self.graph_cache.reset()


def capture_graph(engine :InferenceEngine, decoding_seqlen :int =1, mempool=None, n_warmups :int=3):
    device = engine.model.device
    
    static_input_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, device=device)
    static_position_ids = static_storage_ids.clone().unsqueeze(0)
    # static_position_ids = torch.empty((1, static_storage_ids[0]), dtype=torch.bool, device=device)
    # print(static_position_ids.shape, static_position_ids.shape[-1])
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print("capturing graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids)
    
    def run(input_ids, storage_ids, position_ids):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        graph.replay()
        return static_logits.clone()

    return run

class GraphInferenceEngine:
    def __init__(self, model, cache, graph_cache) -> None:

        self.engine = InferenceEngine(model, cache, graph_cache)
        self.callables = {}
        self.mempool = None
    
    @torch.inference_mode()
    def initialize_cuda_graph(self, n_warmups=3):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        
        self.callables = capture_graph(
            engine=self.engine,
            decoding_seqlen=1,
            mempool=self.mempool,
            n_warmups=3
        )

        self.engine.clear_kv()
    
    @torch.inference_mode()
    def graph_inference(self, input_ids: torch.LongTensor, storage_ids: torch.LongTensor, position_ids: torch.LongTensor):

        # dec_length = input_ids.shape[1]
        logits = self.callables(input_ids, storage_ids, position_ids)
        return logits
    
    def clear_kv(self):
        self.engine.clear_kv()
    
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, storage_ids: Optional[torch.LongTensor]=None):
        return self.engine.model_run(input_ids=input_ids, storage_ids=storage_ids)






