import torch
from torch.utils.data import Dataset


class STDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, device,dataset):
        """
        :param x: shape -> [Batch_size, T, N, C]
        :param y: -> [Batch_size, T, N, C]
        """
        self.x = x.to(device)
        self.y = y.to(device)
        self.dataset = dataset

    def __getitem__(self, item):
        if self.dataset.startswith("PEMS"):
            return self.x[item], self.y[item][:, :, 0].unsqueeze(-1)
        else:
            return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
