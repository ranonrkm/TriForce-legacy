from tqdm import tqdm
import socket
import os

from datasets import load_dataset 
host = socket.gethostname()
if 'lovelace' in host:
    dir_path = "/home/hanshis/workspace/LongContextInfer/data/c4_raw/"
else:
    dir_path = "/fsx-storygen/beidic/hanshi/LongContextInfer/data/c4_raw/"

dataset = load_dataset("c4", "en", split = "train", cache_dir = dir_path)