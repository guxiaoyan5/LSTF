from typing import Tuple

import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset

import controldiffeq
from utils.normalization import StandardScaler
import torchcde


class STDataset(Dataset):
    def __init__(self, coeffs: torch.Tensor, y: torch.Tensor, device):
        # self.coeffs = coeffs.to(device)
        self.coeffs = tuple(b.to(device, dtype=torch.float) for b in coeffs)
        self.y = y.to(device)

    def __getitem__(self, item):
        return tuple(coeffs[item] for coeffs in self.coeffs), self.y[item]

    def __len__(self):
        return len(self.y)


def load_data(dataset: str, root: str = "data") -> np.ndarray:
    data_path = os.path.join(root, dataset, dataset + '.npz')
    if dataset.startswith("PEMS"):
        data = np.load(data_path)['data'][:, :, 0]
    else:
        data = np.load(data_path)['data']
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return data


def generate_sequence(data: np.ndarray, x_offsets: int = 12, y_offsets: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    length = len(data)
    end_index = length - x_offsets - y_offsets + 1
    x = []  # windows
    y = []  # horizon
    index = 0

    while index < end_index:
        x.append(data[index:index + x_offsets])
        y.append(data[index + x_offsets:index + x_offsets + y_offsets])
        index = index + 1
    x = np.array(x)
    y = np.array(y)
    return x, y


def split_data_by_days(data: np.ndarray, val_days: int, test_days: int, interval: int = 5) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    :param data:
    :param val_days:
    :param test_days:
    :param interval: 表示每次采集的时间间隔
    :return:
    """
    t = int((24 * 60) / interval)
    test_data = data[-t * test_days:]
    val_data = data[-t * (test_days + val_days): -t * test_days]
    train_data = data[:-t * (test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data: np.ndarray, val_ratio: float = 0.2, test_ratio: float = 0.2) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def generate_distance(dataset: str, root="data"):
    import pandas
    data = pandas.read_csv(os.path.join(root, dataset, dataset + ".csv"))
    node2id = {}
    if dataset == "PEMS03":
        with open(os.path.join(root, dataset, dataset + ".txt"), 'r', encoding='utf-8') as f:
            nodes = [int(i) for i in f.readlines()]
    else:
        nodes = list(set(data.iloc[:, 0].to_numpy().tolist() + data.iloc[:, 1].to_numpy().tolist()))
        sorted(nodes)
    node2id = {i: index for index, i in enumerate(nodes)}
    num_node = len(nodes)
    dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
    for row in data.itertuples():
        dist_matrix[node2id[row[1]], node2id[row[2]]] = row[3]
        dist_matrix[node2id[row[2]], node2id[row[1]]] = row[3]
    np.save(f'{root}/{dataset}/{dataset}_distance.npy', dist_matrix)


def load_distance(dataset: str, root: str = "data", sigma=None, thres=0.1) -> np.ndarray:
    dist_matrix = np.load(os.path.join(root, dataset, dataset + "_distance.npy"))
    if sigma is None:
        std = np.std(dist_matrix[dist_matrix != np.float('inf')])
        matrix = np.exp(- dist_matrix ** 2 / std ** 2)
    else:
        matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    matrix[matrix < thres] = 0
    return matrix


def get_dataloader_cde(dataset, root="dataset/old/", x_offsets=12, y_offsets=12, batch_size=64, test_ratio=0.1,
                       val_ratio=0.2, device=torch.device('cpu'), missing_test=True, missing_rate=None):
    data = load_data(dataset, root)
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    # add time window
    x_train, y_train = generate_sequence(data_train, x_offsets, y_offsets)
    x_val, y_val = generate_sequence(data_val, x_offsets, y_offsets)
    x_test, y_test = generate_sequence(data_test, x_offsets, y_offsets)
    print('Train: ', x_train.shape, y_train.shape)  # B,T,N,C
    print('Val: ', x_val.shape, y_val.shape)  # B,T,N,C
    print('Test: ', x_test.shape, y_test.shape)  # B,T,N,C
    if missing_test:
        generator = torch.Generator().manual_seed(56789)
        xs = np.concatenate([x_train, x_val, x_test])
        for xi in xs:
            removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[
                                 :int(xs.shape[1] * missing_rate)].sort().values
            removed_points_node = torch.randperm(xs.shape[2], generator=generator)[
                                  :int(xs.shape[2] * missing_rate)].sort().values

            for seq in removed_points_seq:
                for node in removed_points_node:
                    xi[seq, node] = float('nan')
        x_train = xs[:x_train.shape[0], ...]
        x_val = xs[x_train.shape[0]:x_train.shape[0] + x_val.shape[0], ...]
        x_test = xs[-x_test.shape[0]:, ...]
    times = torch.linspace(0, x_offsets - 1, x_offsets)
    augmented_x_train = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_train.shape[0], x_train.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_train[..., :])]
    augmented_x_val = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0], x_val.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_val[..., :])]
    augmented_x_test = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0], x_test.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_test[..., :])]
    x_train = torch.cat(augmented_x_train, dim=3)
    x_val = torch.cat(augmented_x_val, dim=3)
    x_test = torch.cat(augmented_x_test, dim=3)
    # train_coeffs = torchcde.natural_cubic_coeffs(x_train.transpose(1, 2),
    #                                              times)  # # B,N,T,C return torchcde.CubicSpline
    # valid_coeffs = torchcde.natural_cubic_coeffs(x_val.transpose(1, 2), times)
    # test_coeffs = torchcde.natural_cubic_coeffs(x_test.transpose(1, 2), times)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_train.transpose(1, 2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1, 2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1, 2))
    train_dataset = STDataset(train_coeffs, torch.tensor(y_train, dtype=torch.float), device)
    val_dataset = STDataset(valid_coeffs, torch.tensor(y_val, dtype=torch.float), device)
    test_dataset = STDataset(test_coeffs, torch.tensor(y_test, dtype=torch.float), device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader, scaler, times, data.shape[1]
