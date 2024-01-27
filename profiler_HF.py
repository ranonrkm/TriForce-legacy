import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
import torch
import math
from tqdm import tqdm
import time

from models.modeling_llama_cache import LlamaForCausalLM
# from transformers import LlamaForCausalLM
from models.cache_utils import SimpleCache
from torch.profiler import profile, record_function, ProfilerActivity

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')
model = model.eval()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--datalen', type=int, default=1, help='length of data')
    parser.add_argument('--T', type=int, default=2000, help='repeat times')
    args = parser.parse_args()
    
    return args

args = parse_arguments()


data_len = args.datalen
past_key_values = SimpleCache(model, data_len+1200)
from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer)
input_ids = tokenized_prompts[0].to(model.device)[:,:data_len]

T=args.T
LEN = [1]

l = 1
with torch.inference_mode():
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        outputs = model(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values.print_status()

    sentence = torch.randint(low=3, high=30000, size=(1, l)).to(model.device)
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(10):
        outputs = model(
            input_ids=sentence,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values.seq_len -= l
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)

    print(total_time / T, l, data_len, T, "warm up done")

past_key_values.reset()
past_key_values.print_status()
with torch.inference_mode():
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        outputs = model(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values.print_status()

    for l in LEN:
        T = 1000
        sentence = torch.randint(low=3, high=30000, size=(1, l)).to(model.device)
        total_time = 0.0
        torch.cuda.synchronize()
        t1 = time.time()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                for rep in range(T):
                    outputs = model(
                        input_ids=sentence,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                # prof.export_chrome_trace("trace.json")
                    past_key_values.seq_len -= l
                torch.cuda.synchronize()
                t2 = time.time()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
        total_time += (t2 - t1)
        print(total_time / T, l, data_len, T)


