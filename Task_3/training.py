import numpy as np
from network import *
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim


# Training script to train the Recurrent_CNN model for task 2. outputs the model to a file as well as results
if __name__ == "__main__":
    print("Training a Network for IAM Handwriting Database")

    test_ratio = 0.1
    val_ratio = 0.1


    