CUDA_VISIBLE_DEVICES=5 python test/graph_chain_retrieval.py  --gamma 5 --prefill 256 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=7 python test/graph_chain_retrieval.py  --gamma 5 --prefill 512 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/graph_chain_retrieval.py  --gamma 5 --prefill 1024 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=9 python test/graph_chain_retrieval.py  --gamma 5 --prefill 2048 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/graph_chain_retrieval.py  --gamma 5 --prefill 4096 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/graph_chain_retrieval.py  --gamma 5 --prefill 8192 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=7 python test/graph_chain_retrieval.py  --gamma 5 --prefill 16384 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=9 python test/graph_chain_retrieval.py  --gamma 5 --prefill 32768 --budget 0.1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=6 python test/graph_chain_retrieval.py  --gamma 5 --prefill 49152 --budget 0.1 --dataset pg19 --log_csv



CUDA_VISIBLE_DEVICES=1 python test/graph_chain_retrieval.py --verbose --gamma 5 --prefill 32000 --budget 0.05 --dataset pg19 --draft llama-1.1b

CUDA_VISIBLE_DEVICES=7 nohup python test/evict_streamllm.py --prefill 49152 --dataset pg19 --draft llama-1.1b --log_csv --temp 0.8 --draft_cache_budget 2048 > a.log &