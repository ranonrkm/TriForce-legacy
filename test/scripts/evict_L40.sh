CUDA_VISIBLE_DEVICES=1 python test/evict_streamllm.py  --verbose --prefill 256 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=2 python test/evict_streamllm.py  --verbose --prefill 1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=3 python test/evict_streamllm.py  --verbose --prefill 128 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=4 python test/evict_streamllm.py  --verbose --prefill 1024 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=6 python test/evict_streamllm.py  --verbose --prefill 2048 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=7 python test/evict_streamllm.py  --verbose --prefill 4096 --dataset pg19 --log_csv

CUDA_VISIBLE_DEVICES=0 python test/evict_streamllm.py  --verbose --prefill 49152 --dataset pg19 --log_csv

CUDA_VISIBLE_DEVICES=4 python test/evict4evict.py  --verbose --prefill 49152 --dataset pg19 --log_csv
