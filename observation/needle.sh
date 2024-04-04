CUDA_VISIBLE_DEVICES=8 python observation/needle_test.py --budget 4096
CUDA_VISIBLE_DEVICES=9 python observation/needle_test.py --budget 8192
CUDA_VISIBLE_DEVICES=7 python observation/needle_test.py --budget 12288
CUDA_VISIBLE_DEVICES=6 python observation/needle_test.py --budget 16384
CUDA_VISIBLE_DEVICES=5 python observation/needle_test.py --budget 20480
CUDA_VISIBLE_DEVICES=4 python observation/needle_test.py --budget 24576
CUDA_VISIBLE_DEVICES=3 python observation/needle_test.py --budget 28672
CUDA_VISIBLE_DEVICES=2 python observation/needle_test.py --budget 32768
CUDA_VISIBLE_DEVICES=0 python observation/needle_test.py --budget 36864
CUDA_VISIBLE_DEVICES=8 python observation/needle_test.py --budget 40960

CUDA_VISIBLE_DEVICES=7 python observation/needle_test.py --budget 45056
CUDA_VISIBLE_DEVICES=1 python observation/needle_test.py --budget 49152
CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 53248
CUDA_VISIBLE_DEVICES=4,8 python observation/needle_test.py --budget 57344


CUDA_VISIBLE_DEVICES=2,3 python observation/needle_test.py --budget 61440
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 65536
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 69632
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 73728
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 77824
CUDA_VISIBLE_DEVICES=2,3 nohup python observation/needle_test.py --budget 61440 --method h2o > 4.log &
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 86016
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 90112
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 94208
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 98304
CUDA_VISIBLE_DEVICES=8,9 nohup python observation/needle_test.py --budget 102400 > 1.log &
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 106496
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 110592
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 114688
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 118784
# CUDA_VISIBLE_DEVICES=8,9 python observation/needle_test.py --budget 122880
# CUDA_VISIBLE_DEVICES=5,6 python observation/needle_test.py --budget 126976

CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 102400 --method h2o

CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 4096 --method h2o
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 8192 --method h2o
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 16384 --method h2o
CUDA_VISIBLE_DEVICES=0,1 nohup python observation/needle_test.py --budget 32768 --method h2o > h2o1.log &
CUDA_VISIBLE_DEVICES=4,5 nohup python observation/needle_test.py --budget 48128 --method h2o > h2o2.log &
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 65536 --method h2o
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 81920 --method h2o
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 98304 --method h2o
CUDA_VISIBLE_DEVICES=0,1 python observation/needle_test.py --budget 114688 --method h2o