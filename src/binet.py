import torch
import torch.nn as nn


class BiNet(nn.Module):
    def __init__(self, num_classes, context_size=3):
        super(BiNet, self).__init__()

        self.context_size = context_size
        in_channels = context_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=(1, 2),
        )
        n_features = 16 * num_classes
        self.fc1 = nn.Linear(in_features=n_features, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)

        self.letter = nn.Linear(in_features=64, out_features=num_classes)

        self.eow = nn.Linear(in_features=64, out_features=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        pred_letter = self.letter(x)
        pred_eow = self.eow(x)

        pred_letter = self._softmax_temperature(pred_letter, 1)
        pred_eow = torch.sigmoid(pred_eow)

        return pred_letter, pred_eow

    @staticmethod
    def _softmax_temperature(x, temperature=5.0):
        x = x / temperature
        x = torch.softmax(x, dim=1)
        return x

