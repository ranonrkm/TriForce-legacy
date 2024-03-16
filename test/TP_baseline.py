import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch

from models.TP_llama import distributed_init, DistributedLlama
local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)

print(local_rank, world_size, device)

model_name_or_path = "meta-llama/Llama-2-7b-hf"

local_rank, world_size = distributed_init()
device  = torch.device("cuda", local_rank)

llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size)
# for rank in range(world_size):
#     if local_rank == rank:
#         hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
#         llm.init_parameters(hf_model=hf_model)
#         del hf_model
#     dist.barrier()   