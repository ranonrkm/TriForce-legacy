from models.modeling_llama_cache import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
import torch
import time
from tqdm import tqdm
import math
from termcolor import colored
import datetime
from utils.sampling import norm_logits, sample, max_fn

from models.cache_utils import StreamLLMCache, EfficientH2OCache, DejaVuCache

tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", padding_side="left")
model = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="cuda:0")
model = model.eval()

def good_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer)

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--budget', type=float, default=0.1, help='budget of cache')
    parser.add_argument('--cache', type=str, default='h2o', help='cache startegy')
    parser.add_argument('--datalen', type=int, default=32000, help='length of data')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

kv_cache_budget = int(args.budget * args.datalen + args.budget * 200)
datalen = args.datalen

if args.cache == 'h2o':
    past_key_values = EfficientH2OCache(model=model, max_budget=datalen+250, skip_start_layers=1, heavy_size=kv_cache_budget//2, recent_size=kv_cache_budget - kv_cache_budget//2)
elif args.cache == 'streamllm':
    past_key_values = StreamLLMCache(model=model, max_budget=datalen+250, skip_start_layers=1, start_size=16, recent_size=kv_cache_budget - 16)
elif args.cache == 'dejavu':
    past_key_values = DejaVuCache(model=model, max_budget=datalen+250, topk_size=kv_cache_budget, skip_start_layers=-1)
else:
    raise NotImplementedError

for input_ids in tokenized_prompts:
    input_ids = input_ids.to(model.device)[:,:datalen]
    past_key_values.reset()

    top_k = -1
    top_p = 0.9
    temperature = 0.6

    with torch.no_grad():

        iter_prefill = math.ceil(input_ids.shape[1] / 100)
        for i in tqdm(range(iter_prefill)):
            outputs = model(
                input_ids=input_ids[:, i*100:(i+1)*100],
                past_key_values=past_key_values,
                use_cache=True,
                speculation=False,
            )

        past_key_values.print_status()

        print("Prefill length:", past_key_values.seq_len)

        gamma = 4
        verbose = args.verbose

        for i in range(1):
            max_len = 200
            resample_count = 0
            accepted_count = 0
            target_sample_count = 0
            draft_count = 0
            
            past_key_values.print_status()
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

                if isinstance(past_key_values, EfficientH2OCache):
                    past_key_values.update_heavy_cache()

                for _ in range(gamma):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                        speculation=True,
                    )
                    probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
                    pred_token_idx = sample(probs)
                    
                    speculation_probs.append(probs[0])
                    generated_ids.append(pred_token_idx.item())

                    draft_count += 1

                # verification
                verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(model.device)], dim=1)
                past_key_values.seq_len -= gamma

                outputs = model(
                    input_ids=verify_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    speculation=False,
                )

                count = 0
                verify_probs = []
            
                for i in range(gamma + 1):
                    verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

                for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
                    r = torch.rand(1, device = model.device)
                    if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                        count += 1
                        accepted_count += 1
                        n += 1
                        pred_token_idx = torch.tensor([[i]]).to(model.device)
                        if verbose:
                            good_stream(i, tokenizer, 'green')
                    else:
                        resample_count += 1
                        n += 1
                        pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                        if verbose:
                            good_stream(pred_token_idx, tokenizer, 'red')
                        break
                
                if count == len(generated_ids):
                    target_sample_count += 1
                    n += 1
                    pred_token_idx = sample(verify_probs[-1])
                    if verbose:
                        good_stream(pred_token_idx, tokenizer, 'blue')

                next_token = pred_token_idx
                past_key_values.seq_len -= (gamma - count)
            time2 = time.time()
            accepted_rate = accepted_count / draft_count
            avg_tokens = accepted_count / draft_count * gamma
            if verbose:
                print(f"Use {time2 - time1} sec to generate {n} tokens, Tokens/s: {n / (time2 - time1)}", flush=True)
                print(f"generated tokens numbers {n}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
                print(f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
            
            else:
                print(max_len / (time2 - time1), "tokens/s, ", (time2 - time1) / max_len, "s/token", 'Sentence Length:', past_key_values.seq_len, f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")

            # write to file
            with open(f"report/select_{args.cache}.csv", 'a') as f:
                f.write(f"7b-128k,{datalen},{args.budget},{accepted_rate},{(accepted_count)/ draft_count * gamma}\n")