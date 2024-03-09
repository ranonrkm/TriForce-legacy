# our best model
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6

# chunk_size ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 2 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 4 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 16 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 32 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 64 --top_p 0.9 --temp 0.6 --gamma 6

# budget ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 1048 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 2048 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 3072 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 6144 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 8192 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6

# temp ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.2 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.4 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.8 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 1.0 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 1.2 --gamma 6

# top_p ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.5 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.7 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6

# gamma ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 3
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 4
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 5
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 7
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 8
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 9
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 10

# prefill ablation
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 32768 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 49152 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 65536 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 81920 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 98304 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 114688 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
CUDA_VISIBLE_DEVICES=0 python test/e2e_ablation.py --prefill 122880 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6

