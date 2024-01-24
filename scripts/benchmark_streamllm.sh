# python benchmark_cache.py --datalen 1
# python benchmark_cache.py --datalen 1000
# python benchmark_cache.py --datalen 2000
# python benchmark_cache.py --datalen 4000
# python benchmark_cache.py --datalen 8000
# python benchmark_cache.py --datalen 16000
# python benchmark_cache.py --datalen 32000
# python benchmark_cache.py --datalen 50000
# # python benchmark_cache.py --datalen 128000

python benchmark_flash_streamllm.py --datalen 1 --ssl -1
python benchmark_flash_streamllm.py --datalen 1000 --ssl -1
python benchmark_flash_streamllm.py --datalen 2000 --ssl -1
python benchmark_flash_streamllm.py --datalen 4000 --ssl -1
python benchmark_flash_streamllm.py --datalen 8000 --ssl -1
python benchmark_flash_streamllm.py --datalen 16000 --ssl -1
python benchmark_flash_streamllm.py --datalen 32000 --ssl -1
python benchmark_flash_streamllm.py --datalen 50000 --ssl -1

python benchmark_flash_streamllm.py --datalen 1 --ssl 0
python benchmark_flash_streamllm.py --datalen 1000 --ssl 0
python benchmark_flash_streamllm.py --datalen 2000 --ssl 0
python benchmark_flash_streamllm.py --datalen 4000 --ssl 0
python benchmark_flash_streamllm.py --datalen 8000 --ssl 0
python benchmark_flash_streamllm.py --datalen 16000 --ssl 0
python benchmark_flash_streamllm.py --datalen 32000 --ssl 0
python benchmark_flash_streamllm.py --datalen 50000 --ssl 0

python benchmark_flash_streamllm.py --datalen 1 --ssl 1
python benchmark_flash_streamllm.py --datalen 1000 --ssl 1
python benchmark_flash_streamllm.py --datalen 2000 --ssl 1
python benchmark_flash_streamllm.py --datalen 4000 --ssl 1
python benchmark_flash_streamllm.py --datalen 8000 --ssl 1
python benchmark_flash_streamllm.py --datalen 16000 --ssl 1
python benchmark_flash_streamllm.py --datalen 32000 --ssl 1
python benchmark_flash_streamllm.py --datalen 50000 --ssl 1

python benchmark_flash_streamllm.py --datalen 1 --ssl 2
python benchmark_flash_streamllm.py --datalen 1000 --ssl 2
python benchmark_flash_streamllm.py --datalen 2000 --ssl 2
python benchmark_flash_streamllm.py --datalen 4000 --ssl 2
python benchmark_flash_streamllm.py --datalen 8000 --ssl 2
python benchmark_flash_streamllm.py --datalen 16000 --ssl 2
python benchmark_flash_streamllm.py --datalen 32000 --ssl 2
python benchmark_flash_streamllm.py --datalen 50000 --ssl 2

python benchmark_flash_streamllm.py --datalen 1 --ssl 3
python benchmark_flash_streamllm.py --datalen 1000 --ssl 3
python benchmark_flash_streamllm.py --datalen 2000 --ssl 3
python benchmark_flash_streamllm.py --datalen 4000 --ssl 3
python benchmark_flash_streamllm.py --datalen 8000 --ssl 3
python benchmark_flash_streamllm.py --datalen 16000 --ssl 3
python benchmark_flash_streamllm.py --datalen 32000 --ssl 3
python benchmark_flash_streamllm.py --datalen 50000 --ssl 3

python benchmark_flash_streamllm.py --datalen 1 --ssl 4
python benchmark_flash_streamllm.py --datalen 1000 --ssl 4
python benchmark_flash_streamllm.py --datalen 2000 --ssl 4
python benchmark_flash_streamllm.py --datalen 4000 --ssl 4
python benchmark_flash_streamllm.py --datalen 8000 --ssl 4
python benchmark_flash_streamllm.py --datalen 16000 --ssl 4
python benchmark_flash_streamllm.py --datalen 32000 --ssl 4
python benchmark_flash_streamllm.py --datalen 50000 --ssl 4
# python benchmark_flash.py --datalen 128000

# python benchmark_ori.py --datalen 1
# python benchmark_ori.py --datalen 1000
# python benchmark_ori.py --datalen 2000
# python benchmark_ori.py --datalen 4000
# python benchmark_ori.py --datalen 8000
# python benchmark_ori.py --datalen 16000
# python benchmark_ori.py --datalen 32000
# python benchmark_ori.py --datalen 50000
# # python benchmark_flash.py --datalen 128000