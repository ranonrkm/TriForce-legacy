from datasets import load_dataset
from tqdm import tqdm
import secrets
import random
import torch
import json
import os

def get_dataset(dataset_name, tokenizer=None, datalen=None, task=None):
    if dataset_name == '128k':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(len(dataset))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'gs':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(20)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'one-shot':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(1)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    else:
        raise Exception("Dataset not found")