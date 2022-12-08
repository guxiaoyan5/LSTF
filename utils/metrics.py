from typing import Tuple

import numpy as np


def MAE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred - true))


def MSE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    return np.sqrt(MSE(pred, true, mask_value))


def MAPE(pred: np.ndarray, true: np.ndarray, mask_value=None) -> np.ndarray:
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))


def All_Metrics(pred, true, mask1=0.1, mask2=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param pred:
    :param true:
    :param mask1:
    :param mask2: 用来处理MAPE在零值附近导致Inf
    :return:
    """
    assert type(pred) == type(true)
    mae = MAE(pred, true, mask1)
    rmse = RMSE(pred, true, mask1)
    mape = MAPE(pred, true, mask2)

    return mae, rmse, mape
