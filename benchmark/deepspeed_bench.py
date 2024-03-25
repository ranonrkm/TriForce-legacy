import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import perf_counter
import numpy as np
import transformers
import time
# hide generation warnings
transformers.logging.set_verbosity_error()
import argparse


# parser = argparse.ArgumentParser(description='args for main.py')
# parser.add_argument('--ws', type=int, default=1, help='world size')
# parser = deepspeed.add_config_arguments(parser)
# args = parser.parse_args()

def measure_latency(model, tokenizer, payload, generation_args, device):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies = []
    # warm up
    for _ in range(10):
        _ =  model.generate(input_ids, **generation_args)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms

# Model Repository on huggingface.co
# model_id = "NousResearch/Yarn-Llama-2-7b-128k"
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    tensor_parallel={"tp_size": 2},
    dtype=torch.float16,
    )

payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
)
# # generation arguments
# generation_args = dict(do_sample=True, min_length=128, max_new_tokens=128)
# ds_results = measure_latency(ds_model, tokenizer, payload, generation_args, ds_model.module.device)

model = ds_model.module
input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(ds_model.module.device)[:,:1]
output = model(input_ids)

with torch.inference_mode():
    for _ in range(100):
        model(input_ids)
    torch.cuda.synchronize()
    time1 = time.time()
    for _ in range(1000):
        model(input_ids)
    torch.cuda.synchronize()
    print(f"Time taken: {time.time()-time1}")