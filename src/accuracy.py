import numpy as np
import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.correct = 0
        self.total = 0

    def update(self, pred, target):
        pred = torch.argmax(pred, dim=1)
        target = torch.argmax(target, dim=1)
        self.correct += (pred == target).sum()
        self.total += pred.shape[0]

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

