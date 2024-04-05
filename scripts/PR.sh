
# 7B 128K
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 test/TP_offloading_orion.py  --budget 16384 --prefill 130048 --dataset lwm --target lwm-128K --on_chip 6

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 test/TP_offloading_orion.py  --budget 16384 --prefill 130048 --dataset gs --target llama-7B-128K --on_chip 6

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 test/TP_offloading_orion.py  --budget 16384 --prefill 130048 --dataset gs --target lwm-128K-base --on_chip 6

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 test/TP_offloading_orion.py  --budget 10240 --prefill 102400 --dataset gs --target llama-13B-128K --on_chip 0