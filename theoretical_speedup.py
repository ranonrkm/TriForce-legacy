import sys
import pandas as pd


def estimate_time(bsz, num_layers, hidden_size, intermediate_size, seq_len, budget=-1, mlp_projs=2, vocab_size=32000):
    if budget != -1:
        seq_len = min(seq_len, budget)
    
    attn_flops = 2 * bsz * seq_len * hidden_size + 4 * bsz * hidden_size * hidden_size
    attn_mem = 2 * bsz * seq_len * hidden_size + 4 * hidden_size * hidden_size

    mlp_flops = mlp_projs * bsz * hidden_size * intermediate_size
    mlp_mem = mlp_projs * intermediate_size * hidden_size + mlp_projs * bsz * intermediate_size 

    unembedding_flops = bsz * hidden_size * vocab_size
    unembedding_mem = hidden_size * vocab_size 

    attn_compute_time = attn_flops / gpu_tflops_per_sec
    attn_mem_time = attn_mem * 2 / gpu_mem_bandwidth

    mlp_compute_time = mlp_flops / gpu_tflops_per_sec
    mlp_mem_time = mlp_mem * 2 / gpu_mem_bandwidth

    unembedding_compute_time = unembedding_flops / gpu_tflops_per_sec
    unembedding_mem_time = unembedding_mem * 2 / gpu_mem_bandwidth

    total_time = num_layers * (max(attn_compute_time, attn_mem_time) + max(mlp_compute_time, mlp_mem_time)) + max(unembedding_compute_time, unembedding_mem_time)

    compute_time = 0.
    mem_time = 0.
    if attn_mem_time <= attn_compute_time:
        compute_time += attn_compute_time
    if attn_mem_time >= attn_compute_time:
        mem_time += attn_mem_time
    
    if mlp_mem_time <= mlp_compute_time:
        compute_time += mlp_compute_time
    if mlp_mem_time >= mlp_compute_time:
        mem_time += mlp_mem_time

    compute_time = num_layers * compute_time
    mem_time = num_layers * mem_time
    
    if unembedding_mem_time <= unembedding_compute_time:
        compute_time += unembedding_compute_time
    if unembedding_mem_time >= unembedding_compute_time:
        mem_time += unembedding_mem_time

    return compute_time, mem_time, total_time

def speedup(alpha, gamma, c):
    return (1 - alpha**(gamma+1)) / (1 - alpha) / (1 + gamma * c)

if __name__ == '__main__':
    draft_type = sys.argv[1]
    
    base_7b_dict = {
        'num_layers': 32,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'budget': -1,
        'mlp_projs': 3,
        'vocab_size': 32000
    }

    retrieval_7b_dict = {
        'num_layers': 32,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'budget': 512,
        'mlp_projs': 3,
        'vocab_size': 32000
    }

    base_68m_dict = {
        'num_layers': 2,
        'hidden_size': 768,
        'intermediate_size': 3072,
        'budget': -1,
        'mlp_projs': 2,
        'vocab_size': 32000
    }

    streaming_68m_dict = {
        'num_layers': 2,
        'hidden_size': 768,
        'intermediate_size': 3072,
        'budget': 256,
        'mlp_projs': 2,
        'vocab_size': 32000
    }

    if draft_type == 'retrieval':
        draft_dict = retrieval_7b_dict
        alpha_dict = {'2048': 0.73, '4096': 0.684, '8192': 0.632}
    elif draft_type == '68m':
        draft_dict = base_68m_dict
        alpha = 0.5
    elif draft_type == 'streaming_68m':
        draft_dict = streaming_68m_dict
        alpha = 0.1
    else:
        raise NotImplementedError

    vocab_size = 32000 

    gamma=6

    gpu_tflops_per_sec = 312
    gpu_mem_bandwidth = 2

    df = pd.DataFrame(columns=['bsz', 'prefill_len', 'alpha', 'c', 'ratio_draft', 'ratio_target', 'target_total_time', 'target_throughput', 'speedup'])

    '''
    Expectation:
    1. First make sure that batchsize is large enough to make MLP compute bound
    2. Then, if the prefill length is small, then transformer will be compute bound
    3. Then when prefill length increases beyond (#mlp_projs / 2) * intermediate size * (BW_mem / BW_flops), again transformer will become memory bound

    Now, for A100 fp16, 
    BW_flops / BW_mem = 156

    min batchsize required to make MLP compute bound = 156
    Now, use batchsize more than 156, 
    vary prefill length from 2048 to 16384, 
    '''

    for bsz in [1, 4, 16, 64, 256, 1024, 4192]:
        for prefill_len in [2048, 4096, 8192]:
            
            target_compute_time, target_mem_time, target_total_time = estimate_time(bsz=bsz, seq_len=prefill_len, **base_7b_dict)
            draft_compute_time, draft_mem_time, draft_total_time = estimate_time(bsz=bsz, seq_len=prefill_len, **draft_dict)            

            # TODO: update this as in compute-bound regime, verification and generation time might not be the same
            c = draft_total_time / target_total_time    # considering verification takes the same amount of time for target

            alpha = alpha_dict[str(prefill_len)]
            sp = speedup(alpha, gamma, c)
            draft_ratio = draft_compute_time / draft_mem_time
            target_ratio = target_compute_time / target_mem_time

            df.loc[len(df)] = {'bsz': bsz, 'prefill_len': prefill_len, 
                               'alpha': "{:.3f}".format(alpha), 'c': "{:.3f}".format(c),
                               'ratio_draft': "{:.3f}".format(draft_ratio), 
                               'ratio_target': "{:.3f}".format(target_ratio), 
                               'target_total_time': "{:.3f}".format(target_total_time / 1e9),
                               'target_throughput': "{:.3f}".format(bsz * 1e12 / target_total_time),
                               'speedup': "{:2f}".format(sp)}

        df.loc[len(df)] = {'bsz': '', 'prefill_len': '', 'alpha': '', 'c': '', 'ratio_draft': '', 'ratio_target': '', 'target_total_time': '', 'target_throughput': '', 'speedup': ''}

print(df.to_markdown(index=False))