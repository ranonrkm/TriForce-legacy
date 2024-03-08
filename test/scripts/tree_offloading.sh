CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 128
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 256
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 512
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 64
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 4x16
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 8x16
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 32
CUDA_VISIBLE_DEVICES=6 python test/e2e_tree.py --prefill 130752 --budget 0.035 --dataset gs --tree_size 16