from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("emozilla/pg19")
tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m", use_fast=True, legacy=False)

import json
json_file = open('/home/hanshis/workspace/LongContextInfer/data/pg19/pg19-train.json', "a") 
for i in range(28602):
    prompt = dataset['train'][i]['text'][:1024*128*5]
    inpu_ids = tokenizer.encode(prompt, return_tensors='pt')
    if inpu_ids.shape[-1] > 128*1024:
        inpu_ids = inpu_ids[:, :129*1024]
        new_output = tokenizer.decode(inpu_ids[0], skip_special_tokens=True)
        example_data = {
            "text": new_output, 
        }
        json_file.write(json.dumps(example_data) + "\n")

json_file.close()