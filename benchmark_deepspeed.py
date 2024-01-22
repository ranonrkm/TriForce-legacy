import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from models.cache_utils import FlashSimpleCache
from ICML.models.modeling_llama_ori import LlamaForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import deepspeed
import math
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--datalen', type=int, default=128000, help='length of prefill')
parser.add_argument('--T', type=int, default=1000, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
args = parser.parse_args()
print(args)

datalen = 1000

model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map='cuda:0')

from data.dataset import get_dataset
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k")
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')


input_ids = tokenized_prompts[0].to(model.device)[:,:datalen]
past_key_values = None

with torch.no_grad():
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in tqdm(range(iter_prefill)):
        outputs = model(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

model = deepspeed.init_inference(model, dtype=torch.float16, enable_cuda_graph=True)
T = args.T
B = args.B
P = args.P
LEN = [1]


PERFORMANCE = []

for l in LEN:
    sentence = torch.randint(low=3, high=30000, size=(B,  l)).cuda()
    total_time = 0.0
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        output = model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)

    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)
    PERFORMANCE.append(total_time / T)

for i, l in enumerate(LEN):
    print("Length :{}, inference time:{}".format(l, PERFORMANCE[i]))