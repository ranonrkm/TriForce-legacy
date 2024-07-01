from datasets import load_dataset
from tqdm import tqdm
import secrets
import random
import torch
import json
import pickle

# build chat input prompt
def build_chat_input(tokenizer, message):
    # chat format:
    # single-turn: <s>Human: Hello!\n\nAssistant: </s>

    prompt = "<s>Human: " + message + "\n\nAssistant: </s>"

    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    return input_tokens

def build_chat_input_lwm(tokenizer, message, prefill=127*1024):
    # chat format:
    # single-turn: You are a helpful assistant. USER: {} \n ASSISTANT:
    book = tokenizer.encode(message)[:prefill-83]
    prompt = "You are a helpful assistant. USER: Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + tokenizer.decode(book, skip_special_tokens=True) + "\n[end of the book]\n\nNow you have read it. Please summarize it for me. First, tell me the title and the author, and then tell the story in 500 words.\n ASSISTANT:"
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    return input_tokens


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
        dataset = load_dataset("c4", 'en', split='train', streaming=True)
        dataset_head = list(dataset.take(60000))
        c4_idx = json.load(open(f"data/json/c4.json", "r"))['8192']
        # ['4096']
        tokenized_prompts = []
        for i in tqdm(c4_idx):
            prompt = dataset_head[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == '128k':
        import socket
        import os
        # host = socket.gethostname()
        # if 'lovelace' in host:
        #     datasetparent = f"/home/hanshis/workspace/LongContextInfer/data/pg19/"
        # elif 'cr-a100-80-0004' in host:
        #     datasetparent = f"/var/cr06_data/beidic/LongContextInfer/data/pg19/"
        # else:
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
        import socket
        import os
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(20)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts

    elif dataset_name == 'pg19':
        dataset = load_dataset("emozilla/pg19")
        test_valid_dict = json.load(open(f"data/json/pg19.json", "r"))

        if datalen <= 32*1024:
            test_idx = test_valid_dict[str(32*1024)]['test']
            valid_idx = test_valid_dict[str(32*1024)]['valid']
        else:
            test_idx = test_valid_dict[str(128*1024)]['test']
            valid_idx = test_valid_dict[str(128*1024)]['valid']
        
        tokenized_prompts = []

        if datalen < 4096:
            for i in tqdm(test_idx):
                prompt = dataset['test'][i]['text'][:4096*5]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :4096]
                tokenized_prompts.append(tokenized_prompt)
                # assert tokenized_prompt.shape[1] > 120000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"

            for i in tqdm(valid_idx):
                prompt = dataset['validation'][i]['text'][:4096*5]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :4096]
                tokenized_prompts.append(tokenized_prompt)
            return tokenized_prompts
        else:
            for i in tqdm(test_idx):
                prompt = dataset['test'][i]['text'][:datalen*5]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :datalen]
                if datalen == 32*1024:
                    if tokenized_prompt.shape[1] == 32*1024:
                        tokenized_prompts.append(tokenized_prompt)
                else:
                    tokenized_prompts.append(tokenized_prompt)
                # assert tokenized_prompt.shape[1] > 120000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
            
            for i in tqdm(valid_idx):
                prompt = dataset['validation'][i]['text'][:datalen*5]
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :datalen]
                if datalen == 32*1024:
                    if tokenized_prompt.shape[1] == 32*1024:
                        tokenized_prompts.append(tokenized_prompt)
                # assert tokenized_prompt.shape[1] > 120000, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
                else:
                    tokenized_prompts.append(tokenized_prompt)
            return tokenized_prompts
    
    elif dataset_name == 'benchmark':
        dataset = load_dataset("emozilla/pg19")
        test_valid_dict = json.load(open(f"data/json/pg19.json", "r"))

        prompt = dataset['test'][10]['text'][:840000]
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:, :148000]
        assert tokenized_prompt.shape[1] > 128*1024, f"tokenized_prompt.shape[1] = {tokenized_prompt.shape[1]}"
        return [tokenized_prompt]

    elif dataset_name == 'orion':
        dataset = load_dataset("narrativeqa")
        # prompt = build_chat_input(tokenizer, "Please read a part of the book below, and then give me the summary!" + dataset['train'][0]['document']['text'][3:1024*356+527] + "\nNow you have read it! What is the summary?")
        prompt = build_chat_input(tokenizer, "Please read a part of the book below, and then give me the summary!\n[start of the book]\n" + dataset['train'][0]['document']['text'][3:1024*356+473] + "\n[end of the book]\n\nNow you have read it! Please summarize it for me!")
        return [prompt]

    elif dataset_name == 'lwm':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        for i in range(20):
            tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[i]]['document']['text'][3:1024*500])
            if tokenized_prompt.shape[-1] != 127*1024:
                print(i, tokenized_prompt.shape)
                continue
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'password':
        tokenized_prompts = []
        hope_datalen = datalen

        for i in range(100):
            n_garbage = int(3.75 * hope_datalen // 1024 * 1024)
            n_garbage_prefix = n_garbage // 2
            n_garbage_suffix = n_garbage - n_garbage_prefix

            task_description = "You are a helpful assistant. USER: There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
            garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
            garbage_inf = " ".join([garbage] * 15000)
            assert len(garbage_inf) >= n_garbage
            garbage_prefix = garbage_inf[:n_garbage_prefix]
            garbage_suffix = garbage_inf[:n_garbage_suffix]
            pass_key = secrets.token_urlsafe(256)
            # pass_key = ''.join([str(random.randint(0, 9)) for _ in range(256)])
            # print(f"pass_key = {pass_key}")
            information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
            final_question = "What is the pass key? Don't give information outside the document or repeat your findings. Keep your response short and direct. ASSISTANT: The pass key is"
            lines = [
                task_description,
                garbage_prefix,
                information_line,
                garbage_suffix,
            ]
            prompt = "\n".join(lines)

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids[:, :hope_datalen-tokenizer.encode(final_question, return_tensors="pt", add_special_tokens=False).shape[-1]]

            # extend the input_ids to the desired length (input_ids + final_question)
            input_ids = torch.cat([input_ids, tokenizer.encode(final_question, return_tensors="pt", add_special_tokens=False)], dim=-1)

            assert input_ids.shape[-1] == hope_datalen, f"Hope to get a len of {hope_datalen}, but got {input_ids.shape[-1]}"
            # assert only one '1' in the input_ids
            assert torch.sum(input_ids == 1) == 1, f"Expect only one '1' in the input_ids, but got {torch.sum(input_ids == 1)}"

            tokenized_prompts.append(input_ids)
        return tokenized_prompts
    
    elif dataset_name == 'needle_retrieval':
        tokenized_prompts = []
        ans = []
        hope_datalen = datalen
        n_garbage = int(3.75 * hope_datalen // 1024 * 1024)
        for i in tqdm(range(100)):
            n_garbage_prefix = (n_garbage * i) // 100
            n_garbage_suffix = n_garbage - n_garbage_prefix

            task_description = "You are a helpful assistant. USER: There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
            garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
            garbage_inf = " ".join([garbage] * 15000)
            assert len(garbage_inf) >= n_garbage
            garbage_prefix = garbage_inf[:n_garbage_prefix]
            garbage_suffix = garbage_inf[:n_garbage_suffix]
            pass_key = secrets.token_urlsafe(32)[:32]
            ans.append(pass_key)
            # pass_key = ''.join([str(random.randint(0, 9)) for _ in range(256)])
            # print(f"pass_key = {pass_key}")
            information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
            final_question = "What is the pass key? Don't give information outside the document or repeat your findings. Keep your response short and direct. ASSISTANT: The pass key is"
            lines = [
                task_description,
                garbage_prefix,
                information_line,
                garbage_suffix,
            ]
            prompt = "\n".join(lines)

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids[:, :hope_datalen-tokenizer.encode(final_question, return_tensors="pt", add_special_tokens=False).shape[-1]]

            # extend the input_ids to the desired length (input_ids + final_question)
            input_ids = torch.cat([input_ids, tokenizer.encode(final_question, return_tensors="pt", add_special_tokens=False)], dim=-1)

            assert input_ids.shape[-1] == hope_datalen, f"Hope to get a len of {hope_datalen}, but got {input_ids.shape[-1]}"
            # assert only one '1' in the input_ids
            assert torch.sum(input_ids == 1) == 1, f"Expect only one '1' in the input_ids, but got {torch.sum(input_ids == 1)}"

            tokenized_prompts.append(input_ids)
        return tokenized_prompts, ans
    
    elif dataset_name == 'needle_retrieval_cached':
        with open('data/needle_retrieval.pkl', 'rb') as f:
            tokenized_prompts, ans = pickle.load(f)
        return tokenized_prompts, ans
    
    else:
        raise Exception("Dataset not found")