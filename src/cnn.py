import torch
import torch.nn as nn
import torch.nn.functional as func


class ConvNet(nn.Module):
    def __init__(self, img_size, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        features = (img_size - 8)**2
        self.linear = nn.Linear(32*features, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = func.relu(self.linear(x))
        x = func.softmax(self.out(x), dim=1)
        return x

