import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_dfs(path, dataset):
    df_list = []
    for mode in ['-', 'train', 'test']:
        if mode == '-':
            filename = '{}_R.dat'.format(dataset)
        else:
            filename = '{}_{}_R.dat'.format(dataset, mode)
        file = os.path.join(path, dataset, filename)

        if os.path.exists(file):
            df = pd.read_csv(file, sep='\t', index_col=0)
            df_list.append(df.reset_index(drop=True))
    return df_list


def split(num_data, ratio, seed=138):
    shuffled_index = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    index1 = shuffled_index[:int(num_data * ratio)]
    index2 = shuffled_index[int(num_data * ratio):]
    return index1, index2


def normalize(arr):
    avg = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    arr2 = arr - avg
    arr2[:, std != 0] /= std[std != 0]
    return arr2


def read_data(path, dataset, validation):
    if not os.path.exists(os.path.join(path, dataset)):
        raise ValueError(dataset)

    df_list = read_dfs(path, dataset)

    if len(df_list) == 1:
        df = df_list[0]
        arr_x = df.iloc[:, :-1].values.astype(np.float32)
        arr_y = df.iloc[:, -1].values

        trn_idx, test_idx = split(arr_x.shape[0], ratio=0.8)

        trn_x = arr_x[trn_idx]
        trn_y = arr_y[trn_idx]
        test_x = arr_x[test_idx]
        test_y = arr_y[test_idx]

    elif len(df_list) == 2:
        trn_df = df_list[0]
        test_df = df_list[1]

        trn_x = trn_df.iloc[:, :-1].values.astype(np.float32)
        trn_x = normalize(trn_x)
        trn_y = trn_df.iloc[:, -1].values
        test_x = test_df.iloc[:, :-1].values.astype(np.float32)
        test_x = normalize(test_x)
        test_y = test_df.iloc[:, -1].values

    else:
        raise ValueError(dataset)

    nx = trn_x.shape[1]
    ny = trn_y.max() + 1

    if validation:
        trn_idx, val_idx = split(trn_x.shape[0], ratio=0.875)

        val_x = trn_x[val_idx, :]
        val_y = trn_y[val_idx]
        trn_x = trn_x[trn_idx, :]
        trn_y = trn_y[trn_idx]
    else:
        val_x = None
        val_y = None

    return dict(trn_x=trn_x,
                trn_y=trn_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                nx=nx, ny=ny)


def to_loader(x, y, batch_size, shuffle=False):
    x = torch.tensor(x)
    y_type = torch.long if y.dtype == np.int else torch.float
    y = torch.tensor(y, dtype=y_type)
    return DataLoader(TensorDataset(x, y), batch_size, shuffle)
