import os
import torch
import json
from tqdm import tqdm
import argparse
import socket
import warnings
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator

from models.modeling_llama_ori import LlamaForCausalLM

parser = argparse.ArgumentParser() 
parser.add_argument("--file", type=str, default="c4_file0.json", help="json file name")
parser.add_argument("--model", type=str, default="NousResearch/Yarn-Llama-2-7b-128k", help="model name")
parser.add_argument("--prefill", type=int, default=128, help="prefill length")
parser.add_argument("--length", type=int, default=128, help="length of the generated text")
parser.add_argument("--bs", type=int, default=96, help="batch size")

args = parser.parse_args()

host = socket.gethostname()
if 'lovelace' in host:
    datasetparent = "/home/hanshis/workspace/LongContextInfer/data/c4/"
    output_dir = f"/home/hanshis/workspace/LongContextInfer/data/c4-{args.prefill}+{args.length}-{args.model.split('/')[-1]}/"
else:
    datasetparent = "/fsx-storygen/beidic/hanshi/data/c4/"
    output_dir = f"/fsx-storygen/beidic/hanshi/data/c4-{args.prefill}+{args.length}-{args.model.split('/')[-1]}/"

os.makedirs(output_dir, exist_ok=True)

json_file_name = args.file
onedataset = load_dataset("json", data_files = datasetparent + json_file_name, split = "train")

json_file = open(output_dir + json_file_name, "a") 

print(f"loading {json_file_name} from {datasetparent} ==> aligned save to {output_dir + json_file_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model)

def truncate(sample):
    encoded_inputs = tokenizer.encode(sample["text"], truncation=True, max_length=128)
    return {"input_ids": encoded_inputs}

dataset = onedataset.map(truncate, remove_columns=onedataset.column_names, num_proc=32)

def filter_short_samples(example):
    return len(example["input_ids"]) >= 128

dataset = dataset.filter(filter_short_samples, num_proc=32)

train_loader = DataLoader(
    dataset,
    collate_fn=default_data_collator,
    shuffle=True,
    batch_size=args.bs,
)

model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()

tokenizer.pad_token = tokenizer.eos_token

with torch.inference_mode():
    tqdm_bar = tqdm(train_loader, desc="Generating")
    write = 0
    buffer = 0
    
    for batch in train_loader:
        large_outputs = model.generate(input_ids = batch['input_ids'].to(model.device), max_length = 256, do_sample = True, temperature = 0.7, top_p = 0.9, pad_token_id=tokenizer.eos_token_id)
        # assert not torch.any(large_outputs == 2)

        for i in range(large_outputs.shape[0]):
            buffer += 1
            if not torch.any(large_outputs[i] == 2):
                outputs = large_outputs[i] 
                new_output = tokenizer.decode(outputs, skip_special_tokens=True)
                example_data = {
                    "text": new_output, 
                } 
                json_file.write(json.dumps(example_data) + "\n")
                write += 1

                # print(example_data)
                # print(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))
            
        tqdm_bar.set_postfix(write=write, buffer=buffer, ratio=write/buffer)
        tqdm_bar.update(1)
        print(f"write {write} samples / {buffer} samples to {output_dir + json_file_name}")

json_file.close()
print(f"write {write} samples / {buffer} samples to {output_dir + json_file_name}")