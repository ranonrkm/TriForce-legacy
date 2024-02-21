import time
import json
import math
import os
import sys
import torch
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from tqdm import tqdm
import random
from functools import cached_property
import numpy as np
import gcsfs
import tiktoken
from transformers import GenerationConfig, AutoTokenizer
from models.modeling_llama_torch import LlamaForCausalLM


class FLAGS:
    haystack_file="/home/hanshis/workspace/LongContextInfer/data/needle.jsonl"
    max_tokens_per_batch=2000000
    output_file="results.json"
    context_lengths_min=500
    context_lengths_max=520
    n_context_length_intervals=3
    n_document_depth_intervals=3
    n_rounds=2
    seed=1234


class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "You are a helpful assistant. USER: {context} {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. ASSISTANT: "
    RANDOM_NEEDLE_CITIES  = [
        'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
        'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
        'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
        'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
        'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
        'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
        'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
        'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
        'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
        'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
        'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
        'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
    ]

    def __init__(self,
                 needle="HANSHI SUN",
                 haystack_file="/home/hanshis/workspace/LongContextInfer/data/needle.jsonl",
                 retrieval_question="What is the special magic {} number?",
                 results_version = 1,
                 rnd_number_digits = 7,
                 context_lengths_min = 1000,
                 context_lengths_max = 126000,
                 context_lengths_num_intervals = 10,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percent_interval_type = "linear",
                 save_results = False,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True):
        needle="\nThe special magic {city} number is: {rnd_number}\n"
        self.needle = needle
        if not needle or not haystack_file or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        if document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
        elif document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            raise ValueError(f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}")

        self.model = Sampler()

        # self.model.tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        # self.model.tokenizer_tiktoken = tiktoken.encoding_for_model("gpt-4-1106-preview")

    def generate_random_number(self, num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        # print(f"Max context length: {max_context_length}")
        contexts = []
        print(f"Reading jsonl file {self.haystack_file}...")
        with open(self.haystack_file, 'r') as f:
            for _ in range(n):
                context = ""
                toks = 0
                while toks < max_context_length:
                    text = json.loads(f.readline())['text']
                    context += text
                    if toks == 0:
                        toks += len(self.model.tokenizer.encode(text, add_special_tokens = True))
                    else:
                        toks += len(self.model.tokenizer.encode(text, add_special_tokens = False))
                contexts.append(context)
        return contexts

    def encode_and_trim(self, context, context_length):
        tokens = self.model.tokenizer.encode(context)
        if len(tokens) > context_length:
            context = self.model.tokenizer.decode(tokens[:context_length])
        return context

    def create_contexts(self, needle_rnd_number, insert_needle, random_city, trim_context, context_length, depth_percent, seed):
        # if self.save_results:
        #     if self.result_exists(context_length, depth_percent):
        #         return
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        if not insert_needle:
            needle = " " #replace needle with a space
        context = self.generate_context(needle, trim_context, context_length, depth_percent)
        results = {
            'context' : context,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'needle' : needle,
            'question' : question,
            'insert_needle' : insert_needle,
            'needle_rnd_number' : needle_rnd_number,
            'seed': seed,
         }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.model.tokenizer.encode(needle)
        tokens_context = self.model.tokenizer.encode(context)

        # print(f"tokens_needle: {tokens_needle}")
        # print(f"tokens_context: {tokens_context}")

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.model.tokenizer.encode('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.model.tokenizer.decode(tokens_new_context, skip_special_tokens = True)
        return new_context

    def generate_context(self, needle, trim_context, context_length, depth_percent):
        context = self.insert_needle(needle, trim_context, depth_percent, context_length)
        return context

    def compute_max_input_length(self, context_length, buffer=1024):
        return int(context_length)

    def run_test(self):
        fs = gcsfs.GCSFileSystem()
        contexts = []
        template = self.OURS_TEMPLATE

        def _key_from_result(result):
            return (result['context_length'], result['depth_percent'], result['seed'])

        results = []
        full_contexts = self.read_context_files(FLAGS.n_rounds)
        full_tokens = [self.model.tokenizer.encode(full_context) for full_context in tqdm(full_contexts)]
        
        start = time.time()
        for context_length in self.context_lengths:
            trim_contexts = [self.model.tokenizer.decode(full_token[:context_length], add_special_tokens = False) for full_token in tqdm(full_tokens)]
            max_input_length = self.compute_max_input_length(context_length)
            contexts = []
            for depth_percent in self.document_depth_percents:
                for i in range(FLAGS.n_rounds):
                    random_city = random.choice(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES)
                    insert_needle = True
                    needle_rnd_number = str(self.generate_random_number(self.rnd_number_digits))
                    print("context length: " + str(context_length))
                    print("depth_percent : " + str(depth_percent))
                    context = self.create_contexts(needle_rnd_number, insert_needle, random_city, trim_contexts[i], context_length, depth_percent, i)
                    print(context)
                    contexts.append(context)

            if len(contexts) == 0:
                continue

            B = FLAGS.max_tokens_per_batch / (max_input_length + self.model.block_size)
            B = int(B / self.model.data_dim) * self.model.data_dim
            if B < self.model.data_dim:
                B = self.model.data_dim
            elif B > len(contexts):
                B = int(math.ceil(len(contexts) / self.model.data_dim) * self.model.data_dim)
            if len(contexts) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(contexts) % B
            for _ in range(n_pad):
                contexts.insert(0, contexts[0])

            pbar = tqdm(total=len(contexts))
            for i in range(0, len(contexts), B):
                contexts_i = contexts[i:i + B]
                prompts = [
                    template.format(context=context['context'], question=context['question'])
                    for context in contexts_i
                ]
                outs = self.model(prompts, max_input_length)
                for j, (context, out) in enumerate(zip(contexts_i, outs)):
                    if i + j < n_pad:
                        continue
                    results.append({
                        'context_length': context['context_length'],
                        'depth_percent': context['depth_percent'],
                        'response': out,
                        'answer': context['needle_rnd_number'],
                        'correct': context['needle_rnd_number'] in out,
                        'seed': context['seed'],
                    })
                    print(results[-1])

                with open(FLAGS.output_file, 'w') as f:
                    json.dump(results, f)
                pbar.update(len(contexts_i))
            pbar.close()
        print('elapsed', time.time() - start)
        print('done')
                

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()
    


class Sampler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("LargeWorldModel/LWM-Text-1M", use_fast=True, legacy=False)
        self.model = LlamaForCausalLM.from_pretrained("LargeWorldModel/LWM-Text-1M", torch_dtype=torch.float16, device_map="auto")


    def __call__(self, prompts, max_input_length):
        input_ids = self.tokenizer(prompts, return_tensors='pt')['input_ids']

        output = model.generate(
            input_ids=input_ids.to(model.device), max_new_tokens=256, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, num_beams=1, top_k=50, top_p=1.0, temperature=0.0
        ).sequences[:, input_ids.shape[1]:]

        output_text = []
        for text in list(self.tokenizer.batch_decode(output, skip_special_tokens=True)):
            if self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)
        return output_text
        

def main():
    ht = LLMNeedleHaystackTester(
        haystack_file=FLAGS.haystack_file,
        context_lengths_min=FLAGS.context_lengths_min,
        context_lengths_max=FLAGS.context_lengths_max,
        context_lengths_num_intervals=FLAGS.n_context_length_intervals,
        document_depth_percent_intervals=FLAGS.n_document_depth_intervals,
    )
    ht.start_test()

if __name__ == "__main__":
    main()