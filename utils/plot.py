from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_turning_point(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by=['prefill','len'])

    prefill_list = df['prefill'].unique().tolist()
    len_list = df['len'].unique().tolist()

    y = [df[df['prefill'] == prefill]['latency'].tolist() for prefill in prefill_list]

    plt.figure(figsize=(15,10))
    plt.grid()

    k=0
    for i in y:
        plt.plot(len_list,[m / i[0] for m in i], label=f'prefill={prefill_list[k]}')
        k+=1

    plt.ylim(0,6)
    # draw a vertical line at x = 128
    plt.axvline(x=64, color='r', linestyle='--')
    plt.axvline(x=128, color='r', linestyle='--')
    plt.axvline(x=192, color='r', linestyle='--')
    plt.axvline(x=256, color='r', linestyle='--')
    plt.axvline(x=320, color='r', linestyle='--')
    plt.axvline(x=384, color='r', linestyle='--')
    plt.axvline(x=448, color='r', linestyle='--')
    plt.legend()
