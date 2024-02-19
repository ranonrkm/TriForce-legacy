from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import itertools

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
    plt.tight_layout()


def plot_test(file_path, xaxis, title):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(15, 5))
    dims = {}
    dims['dataset'] = df['dataset'].unique()
    spec_args = df.columns[df.columns.get_loc('dataset') + 1:].tolist()
    for spec_arg in spec_args:
        dims[spec_arg] = df[spec_arg].unique()

    x_list = sorted(dims[xaxis])
    # drop 'chunk_size' in dims
    del dims[xaxis]

    params_combos = list(itertools.product(*[dims[param] for param in dims]))
    for combo in params_combos:
        # Building a legend label for the current combination
        label_parts = [f'{param}={value}' for param, value in zip(dims.keys(), combo)]
        label = ', '.join(label_parts)
        print(f"Plotting {label}")
        # Filter dataframe based on the current combination of parameters
        current_filter = [(df[param] == value) for param, value in zip(dims.keys(), combo)]
        if current_filter:  # Check if there are conditions to apply
            current_filter_combined = np.logical_and.reduce(current_filter)
            filtered_df = df[current_filter_combined]
        else:
            filtered_df = df
        
        y_list = []
        for param in x_list:
            subset = filtered_df[filtered_df[xaxis] == param]
            if not subset.empty:
                y_list.append(subset['acceptance_rate'].mean())
        if len(y_list) > 0:
            plt.plot([str(x) for x in x_list], y_list, 'o-', label=label)
        else:
            print(f"No data for {label}")

    # set y-axis limits to 0-1, interval 0.1
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel('Mean Acceptance Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()