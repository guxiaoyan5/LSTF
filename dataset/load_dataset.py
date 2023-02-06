import numpy as np
import os

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import controldiffeq
from dataset import STDataset
from utils.normalization import StandardScaler


def load_data(dataset: str, root: str = "data") -> np.ndarray:
    data_path = os.path.join(root, dataset, dataset + '.npz')
    if dataset.startswith("PEMS") and not dataset.endswith("M"):
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


def get_data_loader(dataset: str = "PEMS08", root: str = "data", x_offsets: int = 12, y_offsets: int = 12,
                    add_time_in_day: bool = True, interval: int = 5, val_ratio: float = 0.2, test_ratio: float = 0.2,
                    batch_size=64, device=torch.device('cpu'), add_day_in_weekday: bool = False):
    """
    :param add_day_in_weekday:
    :param device:
    :param batch_size:
    :param val_ratio:
    :param test_ratio:
    :param dataset:
    :param root:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param interval: 每5分钟间隔
    :return:
    """
    data = load_data(dataset, root)
    num_samples, num_nodes = data.shape[0], data.shape[1]
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    if add_time_in_day:
        t = int((24 * 60) / interval)
        day = num_samples // t + 1
        time_in_day = np.tile(np.tile(np.arange(0, t) / t, [day]), [1, num_nodes, 1]).transpose((2, 1, 0))[:num_samples]
        data = np.concatenate((data, time_in_day), axis=-1)
    if add_day_in_weekday:
        t = int((24 * 60) / interval)
        day = num_samples // t + 1
        weekday = day // 7 + 1
        day_in_weekday = np.tile(np.arange(0, 7) / 7, [t, weekday]).transpose((1, 0)).reshape(-1)
        day_in_weekday = np.tile(day_in_weekday, [1, num_nodes, 1]).transpose((2, 1, 0))[:num_samples]
        data = np.concatenate((data, day_in_weekday), axis=-1)
    train_data, val_data, test_data = split_data_by_ratio(data, val_ratio=val_ratio, test_ratio=test_ratio)
    x_train, y_train = generate_sequence(train_data, x_offsets=x_offsets, y_offsets=y_offsets)
    x_test, y_test = generate_sequence(test_data, x_offsets=x_offsets, y_offsets=y_offsets)
    x_val, y_val = generate_sequence(val_data, x_offsets=x_offsets, y_offsets=y_offsets)
    train_loader = DataLoader(
        STDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float),
                  device=device, dataset=dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        STDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float),
                  device=device, dataset=dataset), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(
        STDataset(torch.tensor(x_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float),
                  device=device, dataset=dataset), batch_size=batch_size, shuffle=False)
    print('mean:', scaler.mean, " std:", scaler.std)
    print('train shape:', train_data.shape)
    print('test shape:', test_data.shape)
    print('val shape:', val_data.shape)
    return train_loader, val_loader, test_loader, scaler, num_nodes


def get_data_loader_cde(dataset: str = "PEMS08", root: str = "data", x_offsets: int = 12, y_offsets: int = 12,
                        add_time_in_day: bool = True, interval: int = 5, val_ratio: float = 0.2,
                        test_ratio: float = 0.2, batch_size=64, device=torch.device('cpu'),
                        add_day_in_weekday: bool = False):
    """
    :param add_day_in_weekday:
    :param device:
    :param batch_size:
    :param val_ratio:
    :param test_ratio:
    :param dataset:
    :param root:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param interval: 每5分钟间隔
    :return:
    """
    data = load_data(dataset, root)
    num_samples, num_nodes = data.shape[0], data.shape[1]
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    if add_time_in_day:
        t = int((24 * 60) / interval)
        day = num_samples // t + 1
        time_in_day = np.tile(np.tile(np.arange(0, t) / t, [day]), [1, num_nodes, 1]).transpose((2, 1, 0))[:num_samples]
        data = np.concatenate((data, time_in_day), axis=-1)
    if add_day_in_weekday:
        t = int((24 * 60) / interval)
        day = num_samples // t + 1
        weekday = day // 7 + 1
        day_in_weekday = np.tile(np.arange(0, 7) / 7, [t, weekday]).transpose((1, 0)).reshape(-1)
        day_in_weekday = np.tile(day_in_weekday, [1, num_nodes, 1]).transpose((2, 1, 0))[:num_samples]
        data = np.concatenate((data, day_in_weekday), axis=-1)
    train_data, val_data, test_data = split_data_by_ratio(data, val_ratio=val_ratio, test_ratio=test_ratio)
    x_train, y_train = generate_sequence(train_data, x_offsets=x_offsets, y_offsets=y_offsets)
    x_test, y_test = generate_sequence(test_data, x_offsets=x_offsets, y_offsets=y_offsets)
    x_val, y_val = generate_sequence(val_data, x_offsets=x_offsets, y_offsets=y_offsets)
    times = torch.linspace(0, 11, 12)
    augmented_X_tra = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_train.shape[0], x_train.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_train[..., :])]
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0], x_val.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_val[..., :])]
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = [
        times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0], x_test.shape[2], 1).unsqueeze(-1).transpose(1, 2),
        torch.Tensor(x_test[..., :])]
    x_test = torch.cat(augmented_X_test, dim=3)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1, 2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1, 2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1, 2))
    train_loader = DataLoader(TensorDataset(*train_coeffs, torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*valid_coeffs, torch.tensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_coeffs, torch.tensor(y_test)), batch_size=batch_size, shuffle=False)
    print('mean:', scaler.mean, " std:", scaler.std)
    print('train shape:', train_data.shape)
    print('test shape:', test_data.shape)
    print('val shape:', val_data.shape)
    return train_loader, val_loader, test_loader, scaler, num_nodes, times
