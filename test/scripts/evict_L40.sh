# CUDA_VISIBLE_DEVICES=1 python test/evict_streamllm.py  --verbose --prefill 256 --dataset pg19 --log_csv
# CUDA_VISIBLE_DEVICES=2 python test/evict_streamllm.py  --verbose --prefill 1 --dataset pg19 --log_csv
# CUDA_VISIBLE_DEVICES=3 python test/evict_streamllm.py  --verbose --prefill 128 --dataset pg19 --log_csv
# CUDA_VISIBLE_DEVICES=4 python test/evict_streamllm.py  --verbose --prefill 1024 --dataset pg19 --log_csv
# CUDA_VISIBLE_DEVICES=6 python test/evict_streamllm.py  --verbose --prefill 2048 --dataset pg19 --log_csv
# CUDA_VISIBLE_DEVICES=7 python test/evict_streamllm.py  --verbose --prefill 4096 --dataset pg19 --log_csv


# CUDA_VISIBLE_DEVICES=1 nohup python test/evict_streamllm.py --prefill 8192 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=5 nohup python test/evict_streamllm.py --prefill 4096 --dataset pg19 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=4 nohup python test/evict_streamllm.py --prefill 16384 --temp 0.8 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=7 nohup python test/evict_streamllm.py --prefill 32768 --dataset pg19 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=7 python test/evict_streamllm.py --prefill 32768 --dataset pg19 --draft llama-1.1b

# CUDA_VISIBLE_DEVICES=0 nohup python test/evict4evict.py --prefill 1 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=0 nohup python test/evict4evict.py --prefill 1 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=1 nohup python test/evict4evict.py --prefill 128 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=1 nohup python test/evict4evict.py --prefill 128 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=0 nohup python test/evict4evict.py --prefill 256 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=0 nohup python test/evict4evict.py --prefill 256 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=3 nohup python test/evict4evict.py --prefill 512 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=3 nohup python test/evict4evict.py --prefill 512 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=4 nohup python test/evict4evict.py --prefill 1024 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=4 nohup python test/evict4evict.py --prefill 1024 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=5 nohup python test/evict4evict.py --prefill 2048 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=5 nohup python test/evict4evict.py --prefill 2048 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=6 nohup python test/evict4evict.py --prefill 4096 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=6 nohup python test/evict4evict.py --prefill 4096 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=7 nohup python test/evict4evict.py --prefill 8192 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=7 nohup python test/evict4evict.py --prefill 8192 --dataset pg19 --temp 0.8 --log_csv > a.log &

# CUDA_VISIBLE_DEVICES=3 nohup python test/evict4evict.py --prefill 32768 --dataset pg19 --log_csv > a.log &
# CUDA_VISIBLE_DEVICES=2 nohup python test/evict4evict.py --prefill 32768 --dataset pg19 --temp 0.8 --log_csv > a.log &


CUDA_VISIBLE_DEVICES=0,1 nohup python test/evict_streamllm.py --prefill 98304 --dataset pg19 --draft llama-1.1b --log_csv --draft_cache_budget 2048 > a.log &
CUDA_VISIBLE_DEVICES=4,5 nohup python test/evict_streamllm.py --prefill 98304 --dataset pg19 --draft llama-1.1b --log_csv --draft_cache_budget 2048 --temp 0.8 > a.log &


CUDA_VISIBLE_DEVICES=0 nohup python test/evict_streamllm.py --prefill 1 --dataset pg19 --draft llama-160m --log_csv > a.log &
CUDA_VISIBLE_DEVICES=0 nohup python test/evict_streamllm.py --prefill 1 --dataset pg19 --draft llama-160m --log_csv --temp 0.8 > a.log &