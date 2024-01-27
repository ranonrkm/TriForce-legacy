CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 32000 --gamma 3 > logs/GS_32k_gamma3.log &
CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 32000 --gamma 4 > logs/GS_32k_gamma4.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 32000 --gamma 5 > logs/GS_32k_gamma5.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 32000 --gamma 6 > logs/GS_32k_gamma6.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 32000 --gamma 7 > logs/GS_32k_gamma7.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 32000 --gamma 8 > logs/GS_32k_gamma8.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 32000 --gamma 9 > logs/GS_32k_gamma9.log &
CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 32000 --gamma 10 > logs/GS_32k_gamma10.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 32000 --gamma 11 > logs/GS_32k_gamma11.log &
CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 32000 --gamma 12 > logs/GS_32k_gamma12.log &

CUDA_VISIBLE_DEVICES=1 python select_flash.py --datalen 32000 --gamma 4 --verbose


CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.05 > logs/GS_32k_budget0.05.log &
CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.055 > logs/GS_32k_budget0.055.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.06 > logs/GS_32k_budget0.06.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.065 > logs/GS_32k_budget0.065.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.07 > logs/GS_32k_budget0.07.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.075 > logs/GS_32k_budget0.075.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.08 > logs/GS_32k_budget0.08.log &
CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.085 > logs/GS_32k_budget0.085.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.09 > logs/GS_32k_budget0.09.log &
CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.095 > logs/GS_32k_budget0.095.log &


CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 32000 --gamma 4 --budget 0.07 --ssl 2 > logs/GS32k_ssl2.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.085 --ssl -1 > logs/GS_32k_ssl1.log &
CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.08 --ssl 0 > logs/GS_32k_ssl01.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.08 --ssl -1 > logs/GS_32k_ssl11.log &


CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 48000 --gamma 3 > logs/GS_32k_gamma3.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 48000 --gamma 4 > logs/GS_32k_gamma4.log &
CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 48000 --gamma 5 > logs/GS_32k_gamma5.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 48000 --gamma 6 > logs/GS_32k_gamma6.log &
CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 48000 --gamma 7 > logs/GS_32k_gamma7.log &
CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 48000 --gamma 8 > logs/GS_32k_gamma8.log &


CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.15 > logs/GS_32k_budget0.15.log &
CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.105 > logs/GS_32k_budget0.055.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.11 > logs/GS_32k_budget0.06.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.115 > logs/GS_32k_budget0.065.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.12 > logs/GS_32k_budget0.07.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.075 > logs/GS_32k_budget0.075.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.08 > logs/GS_32k_budget0.08.log &
CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.085 > logs/GS_32k_budget0.085.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.09 > logs/GS_32k_budget0.09.log &
CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 48000 --gamma 4 --budget 0.095 > logs/GS_32k_budget0.095.log &

CUDA_VISIBLE_DEVICES=2 python select_flash.py --datalen 64000 --gamma 4 > logs/GS_32k_gamma3.log &

CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 56000 --gamma 3 > logs/GS_32k_gamma3.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 56000 --gamma 4 > logs/GS_32k_gamma4.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 56000 --gamma 5 > logs/GS_32k_gamma5.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 56000 --gamma 6 > logs/GS_32k_gamma6.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 56000 --gamma 7 > logs/GS_32k_gamma7.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 56000 --gamma 8 > logs/GS_32k_gamma8.log &

CUDA_VISIBLE_DEVICES=7 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.05 > logs/GS_32k_budget0.05.log &
CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.055 > logs/GS_32k_budget0.055.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.06 > logs/GS_32k_budget0.06.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.065 > logs/GS_32k_budget0.065.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.07 > logs/GS_32k_budget0.07.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.075 > logs/GS_32k_budget0.075.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.08 > logs/GS_32k_budget0.08.log &
CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.085 > logs/GS_32k_budget0.085.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.09 > logs/GS_32k_budget0.09.log &
CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.095 > logs/GS_32k_budget0.095.log &

CUDA_VISIBLE_DEVICES=9 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.085 --ssl 2 > logs/GS32k_ssl211.log &

CUDA_VISIBLE_DEVICES=0 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.085 --ssl 0 > logs/GS32k_ssl2.log &
CUDA_VISIBLE_DEVICES=1 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.085 --ssl -1 > logs/GS_32k_ssl1.log &
CUDA_VISIBLE_DEVICES=2 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.08 --ssl 0 > logs/GS_32k_ssl01.log &
CUDA_VISIBLE_DEVICES=3 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.08 --ssl -1 > logs/GS_32k_ssl11.log &
CUDA_VISIBLE_DEVICES=4 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.075 --ssl 0 > logs/GS_32k_ssl02.log &
CUDA_VISIBLE_DEVICES=5 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.075 --ssl -1 > logs/GS_32k_ssl121.log &
CUDA_VISIBLE_DEVICES=6 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.095 --ssl 0 > logs/GS_32k_ssl03.log &
CUDA_VISIBLE_DEVICES=8 nohup python select_flash.py --datalen 56000 --gamma 4 --budget 0.095 --ssl -1 > logs/GS_32k_ssl13.log &