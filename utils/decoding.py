import torch
import math
import time


import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.misc import spec_stream, log_csv
from utils.sampling import sample, norm_logits, max_fn

@torch.inference_mode()
def Vanilla_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None):
    # reset cache
    target_cache.reset()
    draft_cache.reset()
    
    ############ Iterative Pre-fill ############
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in (range(iter_prefill)):
        outputs = target(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=target_cache,
            use_cache=True,
        )

        outputs_draft = draft(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=draft_cache,
            use_cache=True,
        )

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    n = 0
    time1 = time.time()
    
    ############ Spec Decoding ############
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token

        speculation_probs = []
        generated_ids = []

        for _ in range(gamma):
            outputs = draft(
                input_ids=pred_token_idx,
                past_key_values=draft_cache,
                use_cache=True,
            )

            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            speculation_probs.append(probs[0])
            
            generated_ids.append(pred_token_idx.item())
            draft_count += 1

        # verification
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(draft.device)], dim=1)

        with torch.no_grad():
            outputs = target(
                input_ids=verify_tokens,
                past_key_values=target_cache,
                use_cache=True,
            )

        count = 0
        verify_probs = []
    
        for i in range(gamma + 1):
            assert outputs.logits.shape[1] == gamma + 1
            verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs[:-1]):
            r = torch.rand(1, device = draft.device)

            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(draft.device)
                if verbose:
                    spec_stream(i, tokenizer, 'green')

                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break

            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

        # if eos
        if tokenizer.eos_token_id == pred_token_idx:
            break

        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')

        next_token = pred_token_idx
        
        if gamma - count > 0:
            draft_cache.seq_len -= (gamma - count) - 1
        else:
            # gamma == count, we need to update the cache for draft
            with torch.no_grad():
                outputs = draft(
                    input_ids=torch.tensor([[generated_ids[-1]]]).to(draft.device),
                    past_key_values=draft_cache,
                    use_cache=True,
                )

        target_cache.seq_len -= (gamma - count)
        assert target_cache.seq_len == draft_cache.seq_len, f"{target_cache.seq_len} != {draft_cache.seq_len}"


    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    header = "draft,target,acceptance_rate,token/s,avg_tokens,prefill,gen_len\n"
    entry = f"{draft.config._name_or_path},{target.config._name_or_path},{acceptance_rate},{n / (time2 - time1)},{avg_tokens},{input_ids.shape[1]},{n}\n"

    if file_path is not None:
        log_csv(file_path, header, entry)

    return acceptance_rate


@torch.inference_mode()
def KV_Spec_cache(tokenizer, target, target_cache, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None):
    iter_prefill = math.ceil(input_ids.shape[1] / 100)
    for i in range(iter_prefill):
        outputs = target(
            input_ids=input_ids[:, i*100:(i+1)*100],
            past_key_values=target_cache,
            use_cache=True,
            speculation=False,
        )

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

    n = 0
    time1 = time.time()
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        pred_token_idx = next_token
        
        # speculative decoding
        speculation_probs = []
        generated_ids = []

        if hasattr(target_cache, 'update_cache'):
            target_cache.update_cache()

        for _ in range(gamma):
            outputs = target(
                input_ids=pred_token_idx,
                past_key_values=target_cache,
                use_cache=True,
                speculation=True,
            )
            probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            pred_token_idx = sample(probs)
            
            speculation_probs.append(probs[0])
            generated_ids.append(pred_token_idx.item())

            draft_count += 1

        # verification
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(target.device)], dim=1)
        target_cache.seq_len -= gamma

        outputs = target(
            input_ids=verify_tokens,
            past_key_values=target_cache,
            use_cache=True,
            speculation=False,
        )

        count = 0
        verify_probs = []
    
        for i in range(gamma + 1):
            verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = target.device)
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(target.device)
                if verbose:
                    spec_stream(i, tokenizer, 'green')
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

            if tokenizer.eos_token_id == pred_token_idx:
                break
        
        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')

        next_token = pred_token_idx
        target_cache.seq_len -= (gamma - count)
    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    header = "target,acceptance_rate,token/s,avg_tokens,prefill,gen_len\n"
    entry = f"{target.config._name_or_path},{acceptance_rate},{n / (time2 - time1)},{avg_tokens},{input_ids.shape[1]},{n}\n"

    if file_path is not None:
        log_csv(file_path, header, entry)

@torch.inference_mode()
def Evict_Spec_cache(tokenizer, target, target_cache, draft, draft_cache, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None):
    pass