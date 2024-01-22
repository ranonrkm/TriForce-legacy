import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import time
from tqdm import tqdm
import math
from termcolor import colored
from utils.sampling import norm_logits, sample, max_fn
import copy

from models.cache_utils import SimpleCache
from transformers import AutoTokenizer

from models.modeling_gpt_neox import GPTNeoXForCausalLM
from models.modeling_rwkv import RwkvForCausalLM

# print(f"Model Loaded! {model}, {model_q}")

def good_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--rwkv', type=str, default='430m', help='rwkv model')
    parser.add_argument('--target', type=str, default='gpt-neox-20b', help='target model')
    parser.add_argument('--datalen', type=int, default=1800, help='length of data')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    
    return args

args = parse_arguments()

if args.rwkv == '169m':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile", torch_dtype=torch.float16, device_map="cuda:1")
elif args.rwkv == '430m':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-430m-pile", torch_dtype=torch.float16, device_map="cuda:1")
elif args.rwkv == '1.5b':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-1b5-pile", torch_dtype=torch.float16, device_map="cuda:1")
elif args.rwkv == '3b':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-3b-pile", torch_dtype=torch.float16, device_map="cuda:1")
elif args.rwkv == '7b':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-7b-pile", torch_dtype=torch.float16, device_map="cuda:1")
elif args.rwkv == '14b':
    model_q = RwkvForCausalLM.from_pretrained("RWKV/rwkv-4-14b-pile", torch_dtype=torch.float16, device_map="cuda:1")
else:
    raise NotImplementedError
model_q = model_q.eval()

if args.target == 'gpt-neox-20b':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16, device_map='auto')
elif args.target == 'gpt-neox-20b-8k':
    tokenizer = AutoTokenizer.from_pretrained("kz919/gpt-neox-20b-8k-longtuning")
    model = GPTNeoXForCausalLM.from_pretrained("kz919/gpt-neox-20b-8k-longtuning", torch_dtype=torch.float16, device_map='auto')
elif args.target == 'pythia-6.9b':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", padding_side="left")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.float16, device_map="cuda:0")
else:
    raise NotImplementedError
model = model.eval()


from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name='pg-19', tokenizer=tokenizer, datalen='128k')

datalen = args.datalen

past_key_values = SimpleCache(model=model, max_budget=datalen+250)

for input_ids in tokenized_prompts:
    input_ids = input_ids.to(model.device)[:,:datalen]
    past_key_values.reset()
    past_key_values_q = None

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
            )

            # print(input_ids[:, i*100:(i+1)*100].to(model_q.device), input_ids[:, i*100:(i+1)*100])
            outputs_q = model_q(
                input_ids=input_ids[:, i*100:(i+1)*100].to(model_q.device),
                state=past_key_values_q,
                use_cache=True,
            )

            # assert outputs_q.past_key_values['mha'].seqlen_offset == (i+1)*100, f"got {outputs_q.past_key_values['mha'].seqlen_offset}, but should be {(i+1)*100}"

            past_key_values_q = outputs_q.state

            # print(past_key_values_q[0])
            # exit()

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

                past_key_values_legacy = copy.deepcopy(past_key_values_q)

                for _ in range(gamma):
                    outputs = model_q(
                        input_ids=pred_token_idx.to(model_q.device),
                        state=past_key_values_q,
                        use_cache=True,
                    )

                    # print(past_key_values_q)
                    # print(past_key_values_legacy)

                    probs = norm_logits(outputs.logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p)
                    pred_token_idx = sample(probs)
                    
                    speculation_probs.append(probs[0])
                    generated_ids.append(pred_token_idx.item())

                    past_key_values_q = outputs.state

                    draft_count += 1

                # verification
                verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(model.device)], dim=1)

                outputs = model(
                    input_ids=verify_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                count = 0
                verify_probs = []
            
                for i in range(gamma + 1):
                    verify_probs.append(norm_logits(outputs.logits[:, i, :], temperature=temperature ,top_k=top_k, top_p=top_p)[0])

                for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
                    r = torch.rand(1, device = model.device)
                    if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i].to(verify_prob[i].device))):
                        count += 1
                        accepted_count += 1
                        n += 1
                        pred_token_idx = torch.tensor([[i]]).to(model.device)
                        if verbose:
                            good_stream(i, tokenizer, 'green')
                    else:
                        resample_count += 1
                        n += 1
                        pred_token_idx = sample(max_fn(verify_prob[:50277]-speculation_prob.to(verify_prob.device)))
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

                outputs = model_q(
                    input_ids=verify_tokens[:, :count+1].to(model_q.device),
                    state=past_key_values_legacy,
                    use_cache=True,
                )

                past_key_values_q = outputs.state

                # assert outputs.past_key_values['mha'].key_value_memory_dict[1][0][past_key_values.seq_len][0,0,0] == 0.
                # assert torch.count_nonzero(outputs.past_key_values['mha'].key_value_memory_dict[1][0][past_key_values.seq_len-1]) != 0, f"got {past_key_values.seq_len}, but {outputs.past_key_values['mha'].seqlen_offset}"

            time2 = time.time()
            past_key_values.print_status()
            accepted_rate = accepted_count / draft_count
            avg_tokens = accepted_count / draft_count * gamma
            if verbose:
                print(f"Use {time2 - time1} sec to generate {n} tokens, Tokens/s: {n / (time2 - time1)}", flush=True)
                print(f"generated tokens numbers {n}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
                print(f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
            
            else:
                print(max_len / (time2 - time1), "tokens/s, ", (time2 - time1) / max_len, "s/token", 'Sentence Length:', past_key_values.seq_len, f"accepted rate {accepted_rate}, avg generated tokens {(accepted_count)/ draft_count * gamma}")
