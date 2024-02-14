import os
import socket
from tqdm import tqdm

host = socket.gethostname()
if 'lovelace' in host:
    dir_path = "/home/hanshis/workspace/LongContextInfer/data/c4_raw/downloads/"
else:
    dir_path = "/fsx-storygen/beidic/hanshi/data/c4_raw/downloads/"

files_without_extension = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and '.' not in f]

print("Number of files: ", len(files_without_extension))

if 'lovelace' in host:
    datasetpath = "/home/hanshis/workspace/LongContextInfer/data/c4_raw/downloads/"
    os.makedirs("/home/hanshis/workspace/LongContextInfer/data/c4/", exist_ok=True)
    filtered_dir = "/home/hanshis/workspace/LongContextInfer/data/c4/"
else:
    datasetpath = "/fsx-storygen/beidic/hanshi/data/c4_raw/downloads/"
    os.makedirs("/fsx-storygen/beidic/hanshi/data/c4/", exist_ok=True)
    filtered_dir = "/fsx-storygen/beidic/hanshi/data/c4/"

for i in tqdm(range(len(files_without_extension))): 
    os.system("zcat " + datasetpath + files_without_extension[i] + " > " + filtered_dir + "c4_file{}.json".format(i)) 

print("Done!")