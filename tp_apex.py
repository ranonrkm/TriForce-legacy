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
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

import socket

from contextlib import contextmanager
from typing import Optional
from apex.transformer import parallel_state
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
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    assert rank == local_rank
    process_group = parallel_state.get_tensor_model_parallel_group()
    print(f"Rank {rank} initialized process group with world size {world_size}")
    return rank, world_size, process_group

local_rank, world_size, process_group = init_distributed_environment()

def all_reduce_raw(input_, process_group, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


def cap_graph():
    mempool = torch.cuda.graphs.graph_pool_handle()
    device = torch.device("cuda", local_rank)
    static_hidden_states = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float16)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(1):
            out_states = static_hidden_states + 1
            out_states, _ = all_reduce_raw(out_states, process_group)
        s.synchronize()
        torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    print(out_states)
    print(f"[retrieval run] capturing graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        out_states = static_hidden_states + 1
        out_states, _ = all_reduce_raw(out_states, process_group)
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