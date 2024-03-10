import torch
import math
import time
import numpy as np
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.misc import spec_stream, log_csv
from utils.sampling import sample, norm_logits, max_fn

@torch.inference_mode()
def Baseline(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False):
# reset all cache
    graph_engine.engine.kv_cache.reset()
    logits = graph_engine.inference(input_ids=input_ids)

    if verbose:
        graph_engine.engine.kv_cache.print_status()

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token, tokenizer, 'cyan')

    n = 0
    torch.cuda.synchronize()
    time1 = time.time()
    while n < max_len:
        logits = graph_engine.inference(input_ids=next_token)
        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        n += 1
        if verbose:
            spec_stream(next_token, tokenizer, 'cyan')
    torch.cuda.synchronize()
    time2 = time.time()
    if verbose:
        graph_engine.engine.kv_cache.print_status()
    return (time2 - time1) / n