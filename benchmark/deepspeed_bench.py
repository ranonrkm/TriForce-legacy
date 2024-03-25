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
# hide generation warnings
transformers.logging.set_verbosity_error()


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
model_id = "NousResearch/Yarn-Llama-2-7b-128k"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", trust_remote_code=True, torch_dtype=torch.float16)

payload="Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"*2
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=2,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
print(f"model is loaded on device {ds_model.module.device}")

from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
# assert isinstance(ds_model.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"

payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
    * 2
)
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(do_sample=True, min_length=128, max_new_tokens=128)
ds_results = measure_latency(ds_model, tokenizer, payload, generation_args, ds_model.module.device)

print(f"DeepSpeed model: {ds_results[0]}")