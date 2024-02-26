CUDA_VISIBLE_DEVICES=0 python test/graph_chain_retrieval.py  --gamma 5 --prefill 65536 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=1 python test/graph_chain_retrieval.py  --gamma 5 --prefill 98304 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=2 python test/graph_chain_retrieval.py  --gamma 5 --prefill 114688 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=3 python test/graph_chain_retrieval.py  --gamma 5 --prefill 65536 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=4 python test/graph_chain_retrieval.py  --gamma 5 --prefill 98304 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/graph_chain_retrieval.py  --gamma 5 --prefill 114688 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=6 python test/graph_chain_retrieval.py  --gamma 5 --prefill 65536 --budget 0.05 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=7 python test/graph_chain_retrieval.py  --gamma 5 --prefill 98304 --budget 0.05 --dataset pg19 --log_csv