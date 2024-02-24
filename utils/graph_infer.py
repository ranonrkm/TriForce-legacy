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
    def model_run(self, input_ids: torch.LongTensor, storage_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, gamma_offset: int=0):
        if position_ids is None:
            if input_ids.shape[-1] > 1024: # prefill
                iter_prefill = math.ceil(input_ids.shape[1] / 100)
                for i in tqdm(range(iter_prefill)):
                    logits = self.model(
                        input_ids=input_ids[:, i*100:(i+1)*100],
                        kv_cache=self.kv_cache,
                        graph_cache=None,
                    ).logits
            else: # verification
                logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache).logits
        else: # graph decoding (used for cuda graph capture)
            logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache, storage_ids=storage_ids, position_ids=position_ids, gamma_offset=gamma_offset).logits
        return logits
    
    def clear_kv(self):
        self.graph_cache.reset()


def capture_graph(engine :InferenceEngine, gamma_offset :int =0, mempool=None, n_warmups :int=3):
    device = engine.model.device
    
    static_input_ids = torch.full((1, 1), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(1, device=device)
    static_position_ids = static_storage_ids.clone().unsqueeze(0)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids, gamma_offset=gamma_offset)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print("capturing graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids, gamma_offset=gamma_offset)
    
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
    def initialize_cuda_graph(self, gamma=6):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        
        for gamma_offset in range(gamma):
            self.callables[gamma_offset] = capture_graph(
                                                engine=self.engine,
                                                gamma_offset=gamma_offset,
                                                mempool=self.mempool,
                                                n_warmups=3
                                            )

        self.engine.clear_kv()

    @torch.inference_mode()
    def graph_inference(self, input_ids: torch.LongTensor, storage_ids: torch.LongTensor, position_ids: torch.LongTensor, gamma_offset: int=0):

        logits = self.callables[gamma_offset](input_ids, storage_ids, position_ids)
        return logits
    
    def clear_kv(self):
        self.engine.clear_kv()
    
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, storage_ids: Optional[torch.LongTensor]=None):
        return self.engine.model_run(input_ids=input_ids, storage_ids=storage_ids)
    

    @torch.inference_mode()
    def graph_inference_without_capture(self, input_ids: torch.LongTensor, storage_ids: torch.LongTensor, position_ids: torch.LongTensor, gamma_offset: int=0):
        return self.engine.model_run(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, gamma_offset=gamma_offset)

    def init_graph_cache(self):
        self.engine.graph_cache.init_graph_cache(kv_cache=self.engine.kv_cache)

    def update_graph_cache(self):
        self.engine.graph_cache.update_graph_cache(kv_cache=self.engine.kv_cache)





