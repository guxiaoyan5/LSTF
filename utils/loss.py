import torch
from torch import nn

from utils.normalization import StandardScaler


def scaler_Loss(scaler: StandardScaler, mask_value):
    s = scaler
    l = nn.L1Loss()

    def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        if mask_value is not None:
            mask = torch.gt(y_true, mask_value)
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)
        return l(y_true, y_pred)

    return loss
