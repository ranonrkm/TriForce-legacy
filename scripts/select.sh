# python speculation_select.py --cache h2o --datalen 4000
# python speculation_select.py --cache h2o --datalen 8000
# python speculation_select.py --cache h2o --datalen 16000
# python speculation_select.py --cache h2o --datalen 24000
# python speculation_select.py --cache h2o --datalen 32000

# python speculation_select.py --cache streamllm --datalen 4000
# python speculation_select.py --cache streamllm --datalen 8000
# python speculation_select.py --cache streamllm --datalen 16000
# python speculation_select.py --cache streamllm --datalen 24000
# python speculation_select.py --cache streamllm --datalen 32000

# python speculation_select.py --cache dejavu --datalen 4000
# python speculation_select.py --cache dejavu --datalen 8000
# python speculation_select.py --cache dejavu --datalen 16000
# python speculation_select.py --cache dejavu --datalen 24000
# python speculation_select.py --cache dejavu --datalen 32000

python select_flash.py --datalen 2000 --ssl 0
python select_flash.py --datalen 4000 --ssl 0
python select_flash.py --datalen 8000 --ssl 0
python select_flash.py --datalen 16000 --ssl 0
python select_flash.py --datalen 24000 --ssl 0
python select_flash.py --datalen 32000 --ssl 0
python select_flash.py --datalen 48000 --ssl 0
python select_flash.py --datalen 64000 --ssl 0
python select_flash.py --datalen 80000 --ssl 0
python select_flash.py --datalen 96000 --ssl 0
python select_flash.py --datalen 112000 --ssl 0
python select_flash.py --datalen 128000 --ssl 0

python select_flash.py --datalen 2000 --ssl -1
python select_flash.py --datalen 4000 --ssl -1
python select_flash.py --datalen 8000 --ssl -1
python select_flash.py --datalen 16000 --ssl -1
python select_flash.py --datalen 24000 --ssl -1
python select_flash.py --datalen 32000 --ssl -1
python select_flash.py --datalen 48000 --ssl -1
python select_flash.py --datalen 64000 --ssl -1
python select_flash.py --datalen 80000 --ssl -1
python select_flash.py --datalen 96000 --ssl -1
python select_flash.py --datalen 112000 --ssl -1
python select_flash.py --datalen 128000 --ssl -1