from datasets import load_dataset
from tqdm import tqdm
import json


def get_dataset(dataset_name, tokenizer=None, datalen=None, task=None):
    if dataset_name == "THUDM/LongBench":
        assert task is not None, "task must be specified for THUDM/LongBench"
        dataset = load_dataset("THUDM/LongBench", task, split='test')
        prompt = json.load(open(f"data/json/longbench_prompt.json", "r"))
        prompt_format = prompt[task]

        tokenized_prompts = []
        for i in tqdm(range(len(dataset))):
            json_obj = dataset[i]
            prompt = prompt_format.format(**json_obj)
            prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
            tokenized_prompt = tokenizer.encode(prompt, truncation=False, return_tensors="pt")
            if tokenized_prompt.shape[-1] < 16*1000:
                tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'c4':
        dataset = load_dataset("c4", 'en', split='train', streaming=True, trust_remote_code=True)
        dataset_head = list(dataset.take(60000))
        c4_idx = json.load(open(f"data/json/c4.json", "r"))[datalen]
        tokenized_prompts = []
        for i in tqdm(c4_idx):
            prompt = dataset_head[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'pg19':
        dataset = load_dataset("emozilla/pg19")
        test_valid_dict = json.load(open(f"data/json/pg19.json", "r"))
        test_idx = test_valid_dict[datalen]['test']
        valid_idx = test_valid_dict[datalen]['valid']
        
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
    
    elif dataset_name == 'benchmark':
        dataset = load_dataset("emozilla/pg19")
        test_valid_dict = json.load(open(f"data/json/pg19.json", "r"))

        prompt = dataset['test'][10]['text'][:840000]
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :148000]
        assert tokenized_prompt.shape[1] > 128*1024, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
        return [tokenized_prompt]
    
    else:
        raise Exception("Dataset not found")