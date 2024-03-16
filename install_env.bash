conda create -n TriForce python=3.9
conda activate TriForce
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# e2e test
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 1 --prefill 122880 --gen_len 256 --budget 4096 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 2 --prefill 56320 --gen_len 256 --budget 4096 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 4 --prefill 28160 --gen_len 256 --budget 1024 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 5 --prefill 22528 --gen_len 256 --budget 1024 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 6 --prefill 18944 --gen_len 256 --budget 1024 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 8 --prefill 14080 --gen_len 256 --budget 1024 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 10 --prefill 11264 --gen_len 256 --budget 896 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 12 --prefill 8192 --gen_len 256 --budget 768 --dataset 128k --gamma 5
# CUDA_VISIBLE_DEVICES=0 python test/batch_retrieval.py --bsz 16 --prefill 6656 --gen_len 256 --budget 768  --dataset 128k --gamma 5

# benchmark
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 1 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 2 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 3 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 4 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 5 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 6 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 8 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 10 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 12 --T 1000
CUDA_VISIBLE_DEVICES=0 python benchmark/batch_llama_2.py --bsz 16 --T 1000