import torch
import math
import time
import numpy as np
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.misc import batch_spec_stream, spec_stream
from utils.sampling import sample, norm_logits, max_fn

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / residual.sum(dim=-1, keepdim=True)
    return residual

@torch.inference_mode()
def Baseline(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False):
    graph_engine.engine.kv_cache.reset()
    logits = graph_engine.inference(input_ids=input_ids)
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    if verbose:
        batch_spec_stream(next_token, tokenizer)

    gen_tokens = torch.zeros((input_ids.size(0), max_len), dtype=torch.long, device=input_ids.device)

    n = 0
    torch.cuda.synchronize()
    time1 = time.time()
    while n < max_len:
        # logits = graph_engine.inference(input_ids=next_token)
        logits = graph_engine.engine.model(input_ids=next_token, kv_cache=graph_engine.engine.kv_cache, graph_cache=None).logits
        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        gen_tokens[:, n] = next_token.squeeze()
        n += 1
        if verbose:
            batch_spec_stream(next_token, tokenizer)
    torch.cuda.synchronize()
    time2 = time.time()
    return (time2 - time1) / n, gen_tokens

@torch.inference_mode()
def Retrieval_Spec(tokenizer, graph_engine, input_ids, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None):
    vocab_size = graph_engine.engine.model.config.vocab_size
    graph_engine.clear_kv()
    bsz, prefill = input_ids.size()
    graph_engine.inference(input_ids=input_ids[:,:-1])
    logits = graph_engine.inference(input_ids=input_ids[:,-1:])
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

    n = torch.zeros((bsz,), dtype=torch.long)
    
    resample_count = 0
    bonus_count = 0
    accepted_count = 0
    draft_count = 0
    speculation_probs = torch.zeros((input_ids.size(0), gamma, vocab_size), dtype=torch.float16, device=input_ids.device)
    generated_ids = torch.zeros((input_ids.size(0), gamma), dtype=torch.long, device=input_ids.device)
    accepted_count_list = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)
    
    if verbose:
        gen_tokens = torch.zeros((input_ids.size(0), max_len*2), dtype=torch.long, device=input_ids.device)

    time1 = time.time()
    while n.min() < max_len:
        pred_token_idx = next_token
        
        # speculative decoding

        for gamma_offset in range(gamma):
            position_ids = graph_engine.engine.kv_cache.seq_len[:, None] + gamma_offset
            # logits = graph_engine.engine.model(input_ids=pred_token_idx, gamma_offset=gamma_offset, position_ids=position_ids, kv_cache=graph_engine.engine.kv_cache, graph_cache=graph_engine.engine.graph_cache, spec=True).logits
            # probs = norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
            probs = graph_engine.graph_retrieval_inference(input_ids=pred_token_idx, gamma_offset=gamma_offset, position_ids=position_ids)
            pred_token_idx = sample(probs)
            
            speculation_probs[:, gamma_offset] = probs
            generated_ids[:, gamma_offset] = pred_token_idx.squeeze()

            draft_count += bsz

        # verification
        verify_tokens = torch.cat([next_token, generated_ids], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)
        verify_probs = norm_logits(logits.view(-1, vocab_size), temperature=temperature ,top_k=top_k, top_p=top_p).view(bsz, gamma+1, -1) # (bsz, gamma+1, vocab_size)

        # pre-compute residual for resampling
        # !!!TODO: using cuda graph to speed up sampling
        # residual = (verify_probs[:,:-1] - speculation_probs).relu_() # (bsz, gamma, 32000)
        # residual = residual / (residual.sum(dim=-1, keepdim=True) + 1e-9)
        # resample = sample(residual.view(-1, 32000)).view(bsz, gamma)

        # accept step
        next_token = torch.zeros((bsz,1), dtype=torch.long, device=input_ids.device)
        for j in range(bsz): # bsz
            for i in range(gamma):
                token = generated_ids[j, i]
                spec_prob = speculation_probs[j, i, token]
                verify_prob = verify_probs[j, i, token]
            
                r = torch.rand(1, device = graph_engine.engine.model.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob / spec_prob)):
                    accepted_count_list[j] += 1
                    accepted_count += 1
                    n[j] += 1
                    pred_token_idx = token
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'green')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    # if eos
                    if tokenizer.eos_token_id == token.item():
                        break
                else:
                    resample_count += 1
                    n[j] += 1
                    # pred_token_idx = resample[j, i].unsqueeze(0)
                    pred_token_idx = sample(get_residual(verify_probs[j, i],speculation_probs[j, i]))
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'red')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    break

                if tokenizer.eos_token_id == pred_token_idx.item():
                    break

            if accepted_count_list[j] == gamma:
                bonus_count += 1
                n[j] += 1
                pred_token_idx = sample(verify_probs[j, -1])
                if verbose:
                    # spec_stream(pred_token_idx, tokenizer, 'blue')
                    gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()

            next_token[j] = pred_token_idx.unsqueeze(0)
            # if verbose:
            #     print()
        
        # print(accepted_count_list, n, next_token)

        # rollback kv cache
        graph_engine.engine.kv_cache.seq_len -= (gamma - accepted_count_list)
        graph_engine.update_graph_cache()
        accepted_count_list.zero_()

    time2 = time.time()
    # collect stats
    total_count = accepted_count + resample_count + bonus_count
    avg_tokens = total_count / (draft_count/gamma)
    acceptance_rate = accepted_count / draft_count
    # assert round(acceptance_rate*gamma +1,2)  == round(avg_tokens,2), f"{acceptance_rate*gamma +1} != {avg_tokens}"
    assert total_count == n.sum().item(), f"{total_count} != {n.sum().item()}"
    latency = (time2 - time1) / (total_count / bsz) *1000
    graph_engine.engine.kv_cache.print_status()
    print(f"acceptance rate: {acceptance_rate:.4f} | avg tokens: {avg_tokens:.4f} | latency: {latency:.4f} ms")
    if verbose:
        return acceptance_rate, avg_tokens, latency, gen_tokens
    else:
        return acceptance_rate, avg_tokens, latency, None

@torch.inference_mode()
def Baseline_StreamLLM_Evict(tokenizer, graph_engine, input_ids, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None):
    # reset cache
    graph_engine.clear_kv()
    bsz, prefill = input_ids.size()
    logits = graph_engine.inference(input_ids=input_ids)
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    _ = graph_engine.graph_draft_prefill(input_ids=input_ids)

    n = torch.zeros((bsz,), dtype=torch.long)
    if verbose:
        gen_tokens = torch.zeros((input_ids.size(0), max_len*2), dtype=torch.long, device=input_ids.device)
    resample_count = 0
    bonus_count = 0
    accepted_count = 0
    draft_count = 0
    speculation_probs = torch.zeros((input_ids.size(0), gamma, 32000), dtype=torch.float16, device=input_ids.device)
    generated_ids = torch.zeros((input_ids.size(0), gamma), dtype=torch.long, device=input_ids.device)
    accepted_count_list = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)

    time1 = time.time()
    ############ Spec Decoding ############
    while n.min() < max_len:
        pred_token_idx = next_token
        
        for gamma_offset in range(gamma):
            probs = graph_engine.graph_draft_inference(input_ids=pred_token_idx, gamma_offset = gamma_offset)
            pred_token_idx = sample(probs)
            # print(pred_token_idx)
            speculation_probs[:, gamma_offset] = probs
            generated_ids[:, gamma_offset] = pred_token_idx.squeeze()
            draft_count += bsz

            # verification
        verify_tokens = torch.cat([next_token, generated_ids], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)
        verify_probs = norm_logits(logits.view(-1, 32000), temperature=temperature ,top_k=top_k, top_p=top_p).view(bsz, gamma+1, -1) # (bsz, gamma+1, 32000)

        next_token = torch.zeros((bsz,1), dtype=torch.long, device=input_ids.device)
        bonus_token = torch.zeros((bsz,1), dtype=torch.long, device=input_ids.device)
        for j in range(bsz): # bsz
            for i in range(gamma):
                token = generated_ids[j, i]
                spec_prob = speculation_probs[j, i, token]
                verify_prob = verify_probs[j, i, token]
            
                r = torch.rand(1, device = graph_engine.engine.model.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob / spec_prob)):
                    accepted_count_list[j] += 1
                    accepted_count += 1
                    n[j] += 1
                    pred_token_idx = token
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'green')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    # if eos
                    if tokenizer.eos_token_id == token.item():
                        break
                else:
                    resample_count += 1
                    n[j] += 1
                    # pred_token_idx = resample[j, i].unsqueeze(0)
                    pred_token_idx = sample(get_residual(verify_probs[j, i],speculation_probs[j, i]))
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'red')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    break

                if tokenizer.eos_token_id == pred_token_idx.item():
                    break

            if accepted_count_list[j] == gamma:
                bonus_count += 1
                n[j] += 1
                pred_token_idx = sample(verify_probs[j, -1])
                if verbose:
                    # spec_stream(pred_token_idx, tokenizer, 'blue')
                    gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                bonus_token[j] = verify_tokens[j, -1]

            next_token[j] = pred_token_idx.unsqueeze(0)

        # graph_engine.engine.kv_cache.print_status()
        # print(verify_tokens, accepted_count_list, n, next_token, bonus_token)
        # exit()

        graph_engine.graph_draft_inference(input_ids=bonus_token.to(input_ids.device), gamma_offset = gamma)
        graph_engine.engine.kv_cache.seq_len -= (gamma - accepted_count_list)
        
        current_seq_len = graph_engine.engine.draft_cache.start_size + graph_engine.engine.draft_cache.recent_size + accepted_count_list + 1
        # print(current_seq_len)
        # exit()
        graph_engine.engine.draft_cache.evict_for_spec(current_seq_len)
        accepted_count_list.zero_()

    time2 = time.time()
    # collect stats
    total_count = accepted_count + resample_count + bonus_count
    avg_tokens = total_count / (draft_count/gamma)
    acceptance_rate = accepted_count / draft_count
    # assert round(acceptance_rate*gamma +1,2)  == round(avg_tokens,2), f"{acceptance_rate*gamma +1} != {avg_tokens}"
    assert total_count == n.sum().item(), f"{total_count} != {n.sum().item()}"
    latency = (time2 - time1) / (total_count / bsz) *1000
    # graph_engine.engine.kv_cache.print_status()
    print(f"acceptance rate: {acceptance_rate:.4f} | avg tokens: {avg_tokens:.4f} | latency: {latency:.4f} ms")
    if verbose:
        return acceptance_rate, avg_tokens, latency, gen_tokens
    else:
        return acceptance_rate, avg_tokens, latency


@torch.inference_mode()
def Retrieval_Chain_Spec(tokenizer, graph_engine, input_ids, gamma=6, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None):
    graph_engine.clear_kv()
    bsz = input_ids.size(0)
    graph_engine.inference(input_ids=input_ids[:,:-1])
    logits = graph_engine.inference(input_ids=input_ids[:,-1:])
    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))

    n = torch.zeros((bsz,), dtype=torch.long)
    
    resample_count = 0
    bonus_count = 0
    accepted_count = 0
    draft_count = 0
    speculation_probs = torch.zeros((input_ids.size(0), gamma, 32000), dtype=torch.float16, device=input_ids.device)
    generated_ids = torch.zeros((input_ids.size(0), gamma), dtype=torch.long, device=input_ids.device)
    accepted_count_list = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)
    
    if verbose:
        gen_tokens = torch.zeros((input_ids.size(0), max_len*2), dtype=torch.long, device=input_ids.device)

    time1 = time.time()
    while n.min() < max_len:
        pred_token_idx = next_token
        
        # speculative decoding

        verify_tokens, speculation_probs, acc_rate_middle = Spec_Tiny_for_Middle(next_token, graph_engine, gamma, True, tokenizer)

        # for gamma_offset in range(gamma):
        #     position_ids = graph_engine.engine.kv_cache.seq_len[:, None] + gamma_offset
        #     probs = graph_engine.graph_retrieval_inference(input_ids=pred_token_idx, gamma_offset=gamma_offset, position_ids=position_ids)
        #     pred_token_idx = sample(probs)
            
        #     speculation_probs[:, gamma_offset] = probs
        #     generated_ids[:, gamma_offset] = pred_token_idx.squeeze()

        #     draft_count += bsz

        # verification
        verify_tokens = torch.cat([next_token, generated_ids], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)
        verify_probs = norm_logits(logits.view(-1, 32000), temperature=temperature ,top_k=top_k, top_p=top_p).view(bsz, gamma+1, -1) # (bsz, gamma+1, 32000)

        # pre-compute residual for resampling
        # !!!TODO: using cuda graph to speed up sampling
        # residual = (verify_probs[:,:-1] - speculation_probs).relu_() # (bsz, gamma, 32000)
        # residual = residual / (residual.sum(dim=-1, keepdim=True) + 1e-9)
        # resample = sample(residual.view(-1, 32000)).view(bsz, gamma)

        # accept step
        next_token = torch.zeros((bsz,1), dtype=torch.long, device=input_ids.device)
        for j in range(bsz): # bsz
            for i in range(gamma):
                token = generated_ids[j, i]
                spec_prob = speculation_probs[j, i, token]
                verify_prob = verify_probs[j, i, token]
            
                r = torch.rand(1, device = graph_engine.engine.model.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob / spec_prob)):
                    accepted_count_list[j] += 1
                    accepted_count += 1
                    n[j] += 1
                    pred_token_idx = token
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'green')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    # if eos
                    if tokenizer.eos_token_id == token.item():
                        break
                else:
                    resample_count += 1
                    n[j] += 1
                    # pred_token_idx = resample[j, i].unsqueeze(0)
                    pred_token_idx = sample(max_fn(verify_probs[j, i]-speculation_probs[j, i]))
                    if verbose:
                        # spec_stream(pred_token_idx, tokenizer, 'red')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    break

                if tokenizer.eos_token_id == pred_token_idx.item():
                    break

            if accepted_count_list[j] == gamma:
                bonus_count += 1
                n[j] += 1
                pred_token_idx = sample(verify_probs[j, -1])
                if verbose:
                    # spec_stream(pred_token_idx, tokenizer, 'blue')
                    gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()

            next_token[j] = pred_token_idx.unsqueeze(0)
            # if verbose:
            #     print()
        
        # print(accepted_count_list, n, next_token)

        # rollback kv cache
        graph_engine.engine.kv_cache.seq_len -= (gamma - accepted_count_list)
        graph_engine.update_graph_cache()
        accepted_count_list.zero_()

    time2 = time.time()
    # collect stats
    total_count = accepted_count + resample_count + bonus_count
    avg_tokens = total_count / (draft_count/gamma)
    acceptance_rate = accepted_count / draft_count
    # assert round(acceptance_rate*gamma +1,2)  == round(avg_tokens,2), f"{acceptance_rate*gamma +1} != {avg_tokens}"
    latency = (time2 - time1) / (total_count / bsz) *1000
    graph_engine.engine.kv_cache.print_status()
    print(f"acceptance rate: {acceptance_rate:.4f} | avg tokens: {avg_tokens:.4f} | latency: {latency:.4f} ms")
    if verbose:
        return acceptance_rate, avg_tokens, latency, gen_tokens
    else:
        return acceptance_rate, avg_tokens, latency


def Spec_Tiny_for_Middle(next_token, graph_engine, gamma, verbose, tokenizer):

    bsz = next_token.size(0)
    n = torch.zeros((bsz,), dtype=torch.long)
    
    speculation_probs = torch.zeros((bsz, gamma, 32000), dtype=torch.float16, device=next_token.device)
    return_probs = torch.zeros((bsz, gamma+1, 32000), dtype=torch.float16, device=next_token.device)
    return_ids = torch.zeros((bsz, gamma+1), dtype=torch.long, device=next_token.device)
    accepted_count_list = torch.zeros((bsz,), dtype=torch.long, device=next_token.device)

    pred_token_idx = next_token

    return_ids

    verify_tokens = torch.full((bsz, gamma + 1), 100, device=graph_engine.engine.model.device)
    verify_tokens[:, 0] = next_token

    position_ids = graph_engine.engine.kv_cache.seq_len[:, None] + (gamma+1)

    # time1 = time.time()
    while n.min() < gamma:
        speculation_prob = graph_engine.graph_draft_inference(input_ids=verify_tokens[:,:n+1], gamma_offset = n)
        
        # print(speculation_prob.shape) # (bsz, 32000)
        pred_token_idx = sample(speculation_prob)
        verify_tokens[:, n+1:n+2] = pred_token_idx
        # print(verify_tokens)
        # print(verify_tokens, storage_ids, position_ids)
        verify_prob = graph_engine.graph_verify(input_ids=verify_tokens, position_ids=position_ids)

        # print(verify_prob.shape, speculation_prob.shape)

        r = torch.rand(1, device = graph_engine.engine.model.device)
        # print(n, verify_prob[n, token_idx],  speculation_prob[token_idx])
        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        
        else:
            # pred_token_idx = sample(max_fn(verify_prob[n]-speculation_prob))
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'red')
            resample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx

    # update 68m cache
    graph_engine.graph_draft_inference(input_ids=verify_tokens, gamma_offset = gamma)

    acceptance_rate = accepted_count / draft_count
    return return_generated_ids, return_speculation_probs, acceptance_rate




################### Dist Spec ####################
import torch.distributed as dist

def sample_dist(probs, bsz, tokens):
    if torch.distributed.get_rank() == 0:
        next_token = sample(probs)
    else:
        next_token = torch.empty((bsz, tokens), dtype=torch.long, device=probs.device)

    torch.distributed.broadcast(next_token, src=0)

    return next_token


@torch.inference_mode()
def Baseline_Dist(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, local_rank=0):
    bsz, prefill = input_ids.size()
    graph_engine.reset()
    logits = graph_engine.prefill(input_ids=input_ids)
    
    next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p), bsz, tokens=1)
    
    if verbose:
        batch_spec_stream(next_token, tokenizer)

    gen_tokens = torch.zeros((input_ids.size(0), max_len), dtype=torch.long, device=input_ids.device)

    n = 0
    pos = 0
    generated_ids = []
    generated_ids.extend(next_token[0].tolist())
    
    torch.cuda.synchronize()
    time1 = time.time()
    while n < max_len:
        logits = graph_engine.inference(input_ids=next_token)
        
        next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p), bsz, tokens=1)
        
        generated_ids.extend(next_token[0].tolist())

        generated_text = (
            tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        )
        .strip()
        .split(" ")
        )

        if local_rank == 0:
            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now

        gen_tokens[:, n] = next_token.squeeze()
        n += 1
        if verbose:
            batch_spec_stream(next_token, tokenizer)
    torch.cuda.synchronize()
    time2 = time.time()
    return 1000 * (time2 - time1) / n, gen_tokens

@torch.inference_mode()
def Retrieval_Spec_Dist(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, gamma=6, local_rank=0):
    vocab_size = graph_engine.vocab_size
    graph_engine.reset()
    bsz, prefill = input_ids.size()
    
    graph_engine.prefill(input_ids=input_ids[:,:-1])
    logits = graph_engine.build_retrieval_cache(input_ids=input_ids[:,-1:])
    
    next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p), bsz, tokens=1)

    n = torch.zeros((bsz,), dtype=torch.long)
    
    resample_count = 0
    bonus_count = 0
    accepted_count = 0
    draft_count = 0
    speculation_probs = torch.zeros((input_ids.size(0), gamma, vocab_size), dtype=torch.float16, device=input_ids.device)
    generated_ids = torch.zeros((input_ids.size(0), gamma), dtype=torch.long, device=input_ids.device)
    accepted_count_list = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)
    
    if verbose:
        gen_tokens = torch.zeros((input_ids.size(0), max_len*2), dtype=torch.long, device=input_ids.device)

    time1 = time.time()
    while n.min() < max_len:
        pred_token_idx = next_token
        
        # speculative decoding
        for gamma_offset in range(gamma):
            position_ids = graph_engine.kv_cache.seq_len[:, None] + gamma_offset
            
            probs = graph_engine.retrieval_graph_inference(input_ids=pred_token_idx, gamma_offset=gamma_offset, position_ids=position_ids)
            pred_token_idx = sample_dist(probs, bsz, tokens=1)

            speculation_probs[:, gamma_offset] = probs
            generated_ids[:, gamma_offset] = pred_token_idx.squeeze()

            draft_count += bsz

            dist.barrier()
        # exit()

        # verification
        verify_tokens = torch.cat([next_token, generated_ids], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)
        verify_probs = norm_logits(logits.view(-1, vocab_size), temperature=temperature ,top_k=top_k, top_p=top_p).view(bsz, gamma+1, -1) # (bsz, gamma+1, vocab_size)

        # pre-compute residual for resampling
        # !!!TODO: using cuda graph to speed up sampling
        # residual = (verify_probs[:,:-1] - speculation_probs).relu_() # (bsz, gamma, 32000)
        # residual = residual / (residual.sum(dim=-1, keepdim=True) + 1e-9)
        # resample = sample(residual.view(-1, 32000)).view(bsz, gamma)

        # accept step
        next_token = torch.zeros((bsz,1), dtype=torch.long, device=input_ids.device)
        for j in range(bsz): # bsz
            for i in range(gamma):
                token = generated_ids[j, i]
                spec_prob = speculation_probs[j, i, token]
                verify_prob = verify_probs[j, i, token]
            
                r = torch.rand(1, device = verify_prob.device)
                if r < torch.min(torch.tensor([1], device=r.device), (verify_prob / spec_prob)):
                    accepted_count_list[j] += 1
                    accepted_count += 1
                    n[j] += 1
                    pred_token_idx = token

                    if verbose:
                        if local_rank == 0:
                            spec_stream(pred_token_idx, tokenizer, 'green')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    # if eos
                    if tokenizer.eos_token_id == token.item():
                        break
                else:
                    resample_count += 1
                    n[j] += 1
                    # pred_token_idx = resample[j, i].unsqueeze(0)
                    
                    #!!! NEED REVISE
                    pred_token_idx = sample_dist(get_residual(verify_probs[j, i],speculation_probs[j, i]), bsz=1, tokens=1).view(1)
                    
                    if verbose:
                        if local_rank == 0:
                            spec_stream(pred_token_idx, tokenizer, 'red')
                        gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()
                    break

                if tokenizer.eos_token_id == pred_token_idx.item():
                    break

            if accepted_count_list[j] == gamma:
                bonus_count += 1
                n[j] += 1
                pred_token_idx = sample_dist(verify_probs[j, -1], bsz=1, tokens=1).view(1)

                if verbose:
                    if local_rank == 0:
                        spec_stream(pred_token_idx, tokenizer, 'blue')
                    gen_tokens[j, n[j]-1] = pred_token_idx.squeeze()

            next_token[j] = pred_token_idx.unsqueeze(0)                
            # if verbose:
            #     print()
        
        # print(accepted_count_list, n, next_token)

        # rollback kv cache
        graph_engine.kv_cache.seq_len -= (gamma - accepted_count_list)
        graph_engine.retrieval_cache.update_graph_cache(graph_engine.kv_cache)
        accepted_count_list.zero_()

    time2 = time.time()
    print("rank: ", local_rank)
    # collect stats
    total_count = accepted_count + resample_count + bonus_count
    avg_tokens = total_count / (draft_count/gamma)
    acceptance_rate = accepted_count / draft_count
    # assert round(acceptance_rate*gamma +1,2)  == round(avg_tokens,2), f"{acceptance_rate*gamma +1} != {avg_tokens}"
    assert total_count == n.sum().item(), f"{total_count} != {n.sum().item()}"
    latency = (time2 - time1) / (total_count / bsz) * 1000
    graph_engine.kv_cache.print_status()
    print(f"acceptance rate: {acceptance_rate:.4f} | avg tokens: {avg_tokens:.4f} | latency: {latency:.4f} ms")
    if verbose:
        return acceptance_rate, avg_tokens, latency, gen_tokens
    else:
        return acceptance_rate, avg_tokens, latency, None
