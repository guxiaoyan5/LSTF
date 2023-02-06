import torch
from torch import nn

from utils.normalization import StandardScaler


def scaler_Loss(scaler: StandardScaler):
    s = scaler
    l = nn.L1Loss()

    def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

        return l(y_true, y_pred)

    return loss
