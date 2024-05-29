import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import PIL.Image as Image


def get_dataloaders(data_path, batch_size, val_size=0.2, shuffle=True):
    dataset = MonkBrillDataset(data_path)
    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=shuffle)
    return train_loader, val_loader


class MonkBrillDataset(Dataset):
    """Dataset for MonkBrill Dataset.

    This class loads the characters of the MonkBrill dataset to be used
    for training CNN models.

    TODO:
    * Represent classes equally in validation set.
    """

    def __init__(self, data_path: str) -> None:
        """Constructor for MonkBrillDataset.

        :param data_path: path to the MonkBrill dataset.
        """
        self.data_path = data_path
        self.data = []
        self.label_names = os.listdir(data_path)
        self.labels = []
        for cls_idx, cls in enumerate(self.label_names):
            for item in os.listdir(os.path.join(data_path, cls)):
                self.data.append(os.path.join(data_path, cls, item))
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self._load_img(self.data[idx])
        label = self._one_hot(self.labels[idx], len(self.label_names))
        return img, label

    @staticmethod
    def _load_img(path: str) -> torch.Tensor:
        img = Image.open(path).convert('1')
        img = img.resize((32, 32))
        img = np.array(img, np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        img = 1 - img
        return img

    @staticmethod
    def _one_hot(label: int, num_classes: int = 10) -> torch.Tensor:
        one_hot = torch.zeros(1, num_classes).squeeze(0)
        one_hot[label] = 1
        return one_hot



