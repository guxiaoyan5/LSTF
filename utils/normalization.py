import numpy as np
import torch


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean
