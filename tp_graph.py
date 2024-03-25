import os
import sys

from zmq import device
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
import gc
from termcolor import colored
from utils.batch_decoding import Baseline_Dist, Retrieval_Spec_Dist
from models.TP_llama import distributed_init, DistributedLlama
from models.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import numpy as np
from utils import cupy_utils
from models.TP_llama import distributed_init
import contextlib
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

import socket

from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from utils import custom_all_reduce

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def init_distributed_environment():
    """Initialize the distributed environment."""
    local_rank, world_size = distributed_init()
    cupy_utils.init_process_group(world_size, local_rank)
    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    print(f"Rank {local_rank} initialized process group with world size {world_size}")
    cupy_utils.all_reduce(torch.zeros(1).cuda())
    return local_rank, world_size

local_rank, world_size = init_distributed_environment()

@contextlib.contextmanager
def _maybe_cupy_nccl():
    stream = torch.cuda.current_stream()
    with cupy_utils.set_cupy_stream(stream):
        yield

def cap_graph():
    mempool = torch.cuda.graphs.graph_pool_handle()
    with custom_all_reduce.capture():
        device = torch.device("cuda", local_rank)
        static_hidden_states = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float16)
        with _maybe_cupy_nccl():
            for _ in range(3):
                out_states = static_hidden_states + 1
                cupy_utils.all_reduce(out_states, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        print(out_states)
        print(f"[retrieval run] capturing graph...")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=mempool), _maybe_cupy_nccl():
            out_states = static_hidden_states + 1
            cupy_utils.all_reduce(out_states, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
    def run(hidden_states):
        static_hidden_states.copy_(hidden_states)
        graph.replay()
        return out_states.clone()

    return run

gc.collect()
func = cap_graph()

device = torch.device("cuda", local_rank)
# test cuda graph
hidden_states = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float16)
output = func(hidden_states)
print(hidden_states, output)