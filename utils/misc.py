from sympy import symbols, Eq, solve
from termcolor import colored
import random

def fake2real(fake,gamma=4):
    a = 1+ gamma*fake
    x = symbols('x')
    equation = Eq(x**(gamma+1) - a*x + a - 1, 0)
    solutions = solve(equation, x)
    return solutions[1]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def spec_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

def log_csv(file_path, header, entry):
    try:
        with open(file_path, 'r') as f:
            contents = f.read()
    except FileNotFoundError:
        contents = ""

    if not contents:
        with open(file_path, 'a') as f:
            f.write(header)
    
    with open(file_path, 'a') as f:
        f.write(entry)

def print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path, method, spec_args=None, dataset=None):
    print(colored("####################################### Config #######################################", 'blue'), flush=True)
    print(colored(f"Method: {method}", 'red'), flush=True)
    print(colored(f"Dataset: {dataset}", 'blue'), flush=True)
    print(colored(f"Spec Args: {spec_args}", 'blue'), flush=True)
    print(colored(f"Draft: {draft.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Target: {target.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Prefill Length: {prefill}", 'blue'), flush=True)
    print(colored(f"Generation Length: {gen_len}", 'blue'), flush=True)
    print(colored(f"Gamma: {gamma}", 'blue'), flush=True)
    print(colored(f"Sampling Method: top_k = {top_k}, top_p = {top_p}, temperature = {temperature}", 'blue'), flush=True)
    print(colored(f"Log CSV: {file_path}", 'blue'), flush=True)
    print(colored("######################################################################################\n", 'blue'), flush=True)


import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def rerange_kv_cache(kv_cache, chunk_size):

    num_clusters = kv_cache.seq_len // chunk_size
    assert num_clusters * chunk_size == kv_cache.seq_len, "max_budget should be divisible by chunk_size"
    for layer in tqdm(range(kv_cache.layers)):
        for head_index in range(kv_cache.num_heads):
            # (bsz, max_budget, head_dim) --> (bsz * max_budget, head_dim)
            head_key_cache = kv_cache.key_cache[layer][:, :kv_cache.seq_len, head_index, :].reshape(-1, kv_cache.head_dim).cpu().numpy()
            head_value_cache = kv_cache.value_cache[layer][:, :kv_cache.seq_len, head_index, :].reshape(-1, kv_cache.head_dim).cpu().numpy()
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=head_index).fit(head_key_cache)
            
            labels = kmeans.labels_
            sorted_indices = np.argsort(labels)
            sorted_head_key = head_key_cache[sorted_indices]
            sorted_head_value = head_value_cache[sorted_indices]

        
            kv_cache.key_cache[layer][:, :kv_cache.seq_len, head_index, :] = torch.tensor(sorted_head_key, device=kv_cache.key_cache.device).reshape(1, kv_cache.seq_len, kv_cache.head_dim)
            kv_cache.value_cache[layer][:, :kv_cache.seq_len, head_index, :] = torch.tensor(sorted_head_value, device=kv_cache.key_cache.device).reshape(1, kv_cache.seq_len, kv_cache.head_dim)
    print(f"Rerange KV cache complete, Seq_len: {kv_cache.seq_len}, Chunk_size: {chunk_size}, Clusters: {num_clusters}")