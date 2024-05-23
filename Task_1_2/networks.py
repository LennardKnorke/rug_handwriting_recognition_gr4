import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AugmentAgentCNN(nn.Module):
    def __init__(
        self,
        n_patches: int
    ):
        super(AugmentAgentCNN, self).__init__()
        self.n_patches = n_patches

        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=(3,3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(96, 8*(n_patches+1))

    def forward(
        self, state: np.ndarray
    ):
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.bn2(self.conv5(x)))
        x = self.pool(F.relu(self.bn3(self.conv6(x))))
        x = x.view(-1, 96)
        x = self.fc1(x)
        x = x.view(-1, 2*(self.n_patches+1), 2, 2)
        return x
    
class ClassifierCNN(nn.Module):
    def __init__(
        self,
    ):
        super(ClassifierCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=1, padding=0)

        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))

        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 27)

    def forward(
        self, state: np.ndarray
    ) :
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.pool1(F.relu(self.conv3(x)))

        x = x.view(-1, 1280)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x