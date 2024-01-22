
# KV cache eviction-based speculative decoding
python  speculation_evict.py --verbose  --cache h2o  # extended small model using h2o/streamllm

# KV cache selection-based speculative decoding
python  speculation_select.py --verbose --cache h2o  # 7b-128k our methods (h2o)
python  speculation_select.py --verbose --cache streamllm  # 7b-128k our methods (streamllm)
python  speculation_select.py --verbose --cache dejavu  # 7b-128k our methods (dejavu, upper bound)

# Navie speculative decoding (1.1b-32k for 7b-128k)
python  speculation_naive.py --verbose

# RWKV for gpt-neox-20b or EleutherAI/pythia-6.9b
python  speculation_rnn.py --rwkv 1.5b --target gpt-neox-20b --verbose

# flash decoding for benchmark with Deep Speed
python  benchmark_flash.py

# Dynamic change draft model (h2o, streamingllm, topk) 


# Acceptance rate and speculation length
