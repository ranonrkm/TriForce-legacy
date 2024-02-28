CUDA_VISIBLE_DEVICES=1 python test/recomp_7b.py  --verbose --prefill 1 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=6 python test/recomp_7b.py  --verbose --prefill 256 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=7 python test/recomp_7b.py  --verbose --prefill 512 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=1 python test/recomp_7b.py  --verbose --prefill 2048 --dataset pg19 --log_csv


CUDA_VISIBLE_DEVICES=0 nohup python test/recomp_7b.py  --prefill 2048 --dataset pg19 --log_csv > a.log &
CUDA_VISIBLE_DEVICES=0 nohup python test/recomp_7b.py  --prefill 2048 --dataset pg19 --log_csv --temp 0.8 > a.log &

CUDA_VISIBLE_DEVICES=2 nohup python test/recomp_7b.py  --prefill 8192 --dataset pg19 --log_csv > a.log &
CUDA_VISIBLE_DEVICES=2 nohup python test/recomp_7b.py  --prefill 8192 --dataset pg19 --log_csv --temp 0.8 > a.log &

CUDA_VISIBLE_DEVICES=6 nohup python test/recomp_7b.py  --prefill 16384 --dataset pg19 --log_csv > a.log &
CUDA_VISIBLE_DEVICES=9 nohup python test/recomp_7b.py  --prefill 16384 --dataset pg19 --log_csv --temp 0.8 > a.log &

CUDA_VISIBLE_DEVICES=7 nohup python test/recomp_7b.py  --prefill 32768 --dataset pg19 --log_csv > a.log &
CUDA_VISIBLE_DEVICES=8 nohup python test/recomp_7b.py  --prefill 32768 --dataset pg19 --log_csv --temp 0.8 > a.log &

CUDA_VISIBLE_DEVICES=8 python test/recomp_7b.py  --prefill 49152 --dataset pg19 --log_csv

CUDA_VISIBLE_DEVICES=0 python test/recomp_7b.py  --verbose --prefill 49152 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=1 python test/recomp_7b.py  --verbose --prefill 65536 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=2 python test/recomp_7b.py  --verbose --prefill 98304 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=3 python test/recomp_7b.py  --verbose --prefill 114688 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=4 python test/recomp_7b.py  --verbose --prefill 122880 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/recomp_7b.py  --verbose --prefill 129668 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=4 python test/recomp_7b.py  --verbose --prefill 122880 --dataset pg19 --log_csv
CUDA_VISIBLE_DEVICES=5 python test/recomp_7b.py  --verbose --prefill 129668 --dataset pg19 --log_csv