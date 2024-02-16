from tqdm import tqdm
import socket
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset", type=str, default="OpenOrca", help="dataset name")
args = parser.parse_args()

host = socket.gethostname()
if 'lovelace' in host:
    if args.dataset == "c4":
        dir_path = "/home/hanshis/workspace/LongContextInfer/data/c4_raw/"
        dataset = load_dataset("c4", "en", split = "train", cache_dir = dir_path)
    elif args.dataset == "OpenOrca":
        dir_path = "/home/hanshis/workspace/LongContextInfer/data/OpenOrca_raw/"
        dataset = load_dataset("Open-Orca/OpenOrca", split = "train", cache_dir = dir_path)
else:
    if args.dataset == "c4":
        dir_path = "/fsx-storygen/beidic/hanshi/data/c4_raw/"
        dataset = load_dataset("c4", "en", split = "train", cache_dir = dir_path)
    elif args.dataset == "OpenOrca":
        dir_path = "/fsx-storygen/beidic/hanshi/data/OpenOrca_raw/"
        dataset = load_dataset("Open-Orca/OpenOrca", split = "train", cache_dir = dir_path)