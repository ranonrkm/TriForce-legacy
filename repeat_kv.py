import torch
from flash_attn import flash_attn_with_kvcache
from transformers.models.llama.modeling_llama import(
    repeat_kv,
)
import torch.nn.functional as F
import math
import time

def benchmark_mqa_attn(attn_method, query_states, key_states, value_states, num_key_value_groups, seq_len):
    bsz, kv_len, kv_heads, head_dim = key_states.shape
    if attn_method == 'flash':
        # warm up
        for i in range(100):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, cache_seqlens=seq_len, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
        torch.cuda.synchronize()
        T = 2000
        time1 = time.time()
        for i in range(T):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            attn_output = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, cache_seqlens=seq_len, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
        torch.cuda.synchronize()
        latency = (time.time()-time1) / T * 1000

    elif attn_method == 'flash-ref':
        # warm up
        for i in range(100):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            flash_attn_with_kvcache(q=query_states[:,:,:4], k_cache=key_states, v_cache=value_states, cache_seqlens=seq_len, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
        torch.cuda.synchronize()
        T = 2000
        time1 = time.time()
        for i in range(T):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            attn_output = flash_attn_with_kvcache(q=query_states[:,:,:4], k_cache=key_states, v_cache=value_states, cache_seqlens=seq_len, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
        torch.cuda.synchronize()
        latency = (time.time()-time1) / T * 1000
    
    elif attn_method == 'vanilla':
        bsz, kv_len, kv_heads, head_dim = key_states.shape
        # warm up
        for i in range(100):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            key_states_cp = repeat_kv(key_states.transpose(1,2), num_key_value_groups).transpose(1,2)
            value_states_cp = repeat_kv(value_states.transpose(1,2), num_key_value_groups).transpose(1,2)
            attn_weights = torch.matmul(query_states.transpose(1,2), key_states_cp.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_cp.transpose(1,2))
            attn_output = attn_output.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        T = 2000
        time1 = time.time()
        for i in range(T):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            key_states_cp = repeat_kv(key_states.transpose(1,2), num_key_value_groups).transpose(1,2)
            value_states_cp = repeat_kv(value_states.transpose(1,2), num_key_value_groups).transpose(1,2)
            attn_weights = torch.matmul(query_states.transpose(1,2), key_states_cp.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_cp.transpose(1,2))
            attn_output = attn_output.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        latency = (time.time()-time1) / T * 1000

    elif attn_method == 'vanilla-ref':
        bsz, kv_len, kv_heads, head_dim = key_states.shape
        # warm up
        for i in range(100):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            attn_weights = torch.matmul(query_states[:,:,:4].transpose(1,2), key_states.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states.transpose(1,2))
            attn_output = attn_output.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        T = 2000
        time1 = time.time()
        for i in range(T):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            attn_weights = torch.matmul(query_states[:,:,:4].transpose(1,2), key_states.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states.transpose(1,2))
            attn_output = attn_output.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        latency = (time.time()-time1) / T * 1000

    elif attn_method == 'optim':
        bsz, kv_len, kv_heads, head_dim = key_states.shape
        _, query_len, _, _ = query_states.shape
        for i in range(100):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            query_states_cp = query_states.transpose(1,2).reshape(bsz, kv_heads, num_key_value_groups*query_len, head_dim)
            attn_weights = torch.matmul(query_states_cp, key_states.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim) # [bsz, 4, 8*seq, prefill+seq]
            # [TODO] add attn mask here....
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states.transpose(1,2)) # [bsz, 4, 8*seq, 128]
            attn_output = attn_output.reshape(bsz, kv_heads*num_key_value_groups, query_len, head_dim).transpose(1, 2).contiguous() # [bsz, seq, 32, 128]
        torch.cuda.synchronize()
        T = 2000
        time1 = time.time()
        for i in range(T):
            query_states.normal_()
            key_states.normal_()
            value_states.normal_()
            query_states_cp = query_states.transpose(1,2).reshape(bsz, kv_heads, num_key_value_groups*query_len, head_dim)
            attn_weights = torch.matmul(query_states_cp, key_states.transpose(1,2).transpose(2, 3)) / math.sqrt(head_dim) # [bsz, 4, 8*seq, prefill+seq]
            # [TODO] add attn mask here....
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states.transpose(1,2)) # [bsz, 4, 8*seq, 128]
            attn_output = attn_output.reshape(bsz, kv_heads*num_key_value_groups, query_len, head_dim).transpose(1, 2).contiguous() # [bsz, seq, 32, 128]
        torch.cuda.synchronize()
        latency = (time.time()-time1) / T * 1000

    return latency


num_key_value_groups = 8
prefill = 1024*64

# bsz_list = [1,2,3,4,5,6,7,8]
bsz_list = [1,2,4,8]
input_list = [1,2,4,8,16]

for bsz in bsz_list:
    for input in input_list:
        seq_len = torch.tensor([prefill]*bsz, dtype=torch.int32, device="cuda:0")
        query_states = torch.randn(bsz, input, 32, 128, dtype=torch.float16, device="cuda:0")
        key_states = torch.randn(bsz, prefill+input, 32//num_key_value_groups, 128, dtype=torch.float16, device="cuda:0")
        value_states = torch.randn(bsz, prefill+input, 32//num_key_value_groups, 128, dtype=torch.float16, device="cuda:0")
        latency = benchmark_mqa_attn('flash', query_states, key_states, value_states, num_key_value_groups, seq_len)
        print(f"bsz: {bsz}, prefill: {seq_len.cpu().numpy()}, input: {input}, flash: {latency}")
        latency = benchmark_mqa_attn('vanilla', query_states, key_states, value_states, num_key_value_groups, seq_len)
        print(f"bsz: {bsz}, prefill: {seq_len.cpu().numpy()}, input: {input}, vanilla: {latency}")
        latency = benchmark_mqa_attn('optim', query_states, key_states, value_states, num_key_value_groups, seq_len)
        print(f"bsz: {bsz}, prefill: {seq_len.cpu().numpy()}, input: {input}, optim: {latency}")
        latency = benchmark_mqa_attn('flash-ref', query_states, key_states, value_states, num_key_value_groups, seq_len)
        print(f"bsz: {bsz}, prefill: {seq_len.cpu().numpy()}, input: {input}, flash-ref: {latency}")
        latency = benchmark_mqa_attn('vanilla-ref', query_states, key_states, value_states, num_key_value_groups, seq_len)
        print(f"bsz: {bsz}, prefill: {seq_len.cpu().numpy()}, input: {input}, vanilla-ref: {latency}")
        print("=======================================================================")
    print("***********************************************************************")