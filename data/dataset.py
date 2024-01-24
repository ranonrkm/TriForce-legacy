from datasets import load_dataset
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

import json


def get_dataset(dataset_name, split="test", tokenizer=None, batch_size=1, datalen=None, task=None):

    if dataset_name == "alespalla/chatbot_instruction_prompts":
        dataset = load_dataset(dataset_name, split=split)
        if batch_size == 1:
            tokenized_prompts = []
            for prompt in tqdm(dataset["prompt"][:100]):
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                tokenized_prompts.append(tokenized_prompt)
            return tokenized_prompts

        else:
            def convert_to_tensor(batch):
                #TODO: For LLaMA, we do not have a pad_token, so we use [PAD] instead
                tokenizer.pad_token = "PAD"
                inputs = tokenizer(batch['prompt'], padding='max_length', truncation=True, max_length=100, return_tensors="pt")
                # assert type(inputs['input_ids']) == torch.Tensor
                batch['input_ids'] = inputs['input_ids']
                batch['attention_mask'] = inputs['attention_mask']
                batch.pop('response', None)
                batch.pop('prompt', None)
                return batch
            tensor_dataset = dataset.map(convert_to_tensor)
            
            def collate_fn(batch):
                input_ids = torch.stack([torch.tensor(item['input_ids']).squeeze(0) for item in batch])
                attention_mask = torch.stack([torch.tensor(item['attention_mask']).squeeze(0) for item in batch])
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
            
            dataloader = DataLoader(tensor_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    elif dataset_name == "ccdv/cnn_dailymail":
        dataset = load_dataset(dataset_name, '3.0.0', split=split)
        if batch_size == 1:
            if datalen == '1.5k':
                import pickle
                idx = pickle.load(open('data/cnn_1.5k_idx.pkl', 'rb'))
                tokenized_prompts = []
                for i in tqdm(idx[:100]):
                    prompt = dataset[i]['article']
                    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                    tokenized_prompts.append(tokenized_prompt)
                
                return tokenized_prompts
            
            elif datalen == 'full':
                tokenized_prompts = []
                for prompt in tqdm(dataset["article"]):
                    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                    tokenized_prompts.append(tokenized_prompt)
                return tokenized_prompts
            
            else:
                tokenized_prompts = []
                for prompt in tqdm(dataset["article"][:100]):
                    # tokenized_prompt = tokenizer.encode('You are given a report by CNN:' + prompt + '\n\nNow, give me some highlights of this report in your own words, be careful not to repeat them.\n\n', return_tensors="pt")
                    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                    tokenized_prompts.append(tokenized_prompt)
                return tokenized_prompts

    elif dataset_name == "THUDM/LongBench":
        dataset = load_dataset("THUDM/LongBench", task)

        if datalen is None:
            print(f"Datalen not specified!", mode="F")
            exit()
        elif task is None:
            print(f"Task not specified!", mode="F")
            exit()
        else:
            print(f"Loading dataset {dataset_name}.{task} with datalen {datalen} ...")
            idx = json.load(open(f"data/longbench_{task}.json", "r"))
            idx = idx[datalen]
        
        prompt = json.load(open(f"data/longbench_prompt.json", "r"))
        prompt_format = prompt[task]
        
        if batch_size == 1:
            tokenized_prompts = []
            for i in tqdm(idx):
                json_obj = dataset['test'][i]
                prompt = prompt_format.format(**json_obj)
                prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
                tokenized_prompt = tokenizer.encode(prompt, truncation=False, return_tensors="pt")
                tokenized_prompts.append(tokenized_prompt)
            return tokenized_prompts

    elif dataset_name == 'c4':
        dataset = load_dataset("c4", 'en', split='train', streaming=True)
        dataset_head = list(dataset.take(100))
        tokenized_prompts = []
        for prompt in tqdm(dataset_head):
            tokenized_prompt = tokenizer.encode(prompt['text'], return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'pg-19':
        if datalen == '128k':
            dataset = load_dataset("emozilla/pg19")
            test_valid_dict = json.load(open(f"data/pg-19-128k.json", "r"))
            test_idx = test_valid_dict['test']
            valid_idx = test_valid_dict['valid']

            tokenized_prompts = []
            for i in tqdm(test_idx):
                prompt = dataset['test'][i]['text'][:840000]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :128000]
                tokenized_prompts.append(tokenized_prompt)
                assert tokenized_prompt.shape[1] > 120000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
            for i in tqdm(valid_idx):
                prompt = dataset['validation'][i]['text'][:840000]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :128000]
                tokenized_prompts.append(tokenized_prompt)
                assert tokenized_prompt.shape[1] > 120000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
            return tokenized_prompts
        else:
            dataset = load_dataset("emozilla/pg19")
            test_valid_dict = json.load(open(f"data/pg-19-32k.json", "r"))
            test_idx = test_valid_dict['test'][:1]
            valid_idx = test_valid_dict['valid'][:1]

            tokenized_prompts = []
            for i in tqdm(test_idx):
                prompt = dataset['test'][i]['text'][:400000]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :32000]
                tokenized_prompts.append(tokenized_prompt)
                assert tokenized_prompt.shape[1] >= 31000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
            for i in tqdm(valid_idx):
                prompt = dataset['validation'][i]['text'][:400000]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :32000]
                tokenized_prompts.append(tokenized_prompt)
                assert tokenized_prompt.shape[1] >= 31000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
            return tokenized_prompts
    
    else:
        raise Exception("Dataset not found")

    return dataloader