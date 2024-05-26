import cv2
import numpy as np
from network import *
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim

from typing import Tuple

def get_y_from_file(name : str, labelFile) -> str:
    """
    Search for name in list and return line below it

    """
    for i, row in enumerate(labelFile):
        if name == row:
            return labelFile[i + 1]
    return ""

def load_data(paths : list, file_names : list, labelFile)-> Tuple[np.ndarray, list, list, list]:
    """
    Create the x, y dataset as well as the file names
    Load the images as x, and read the target sentence from the label file
    @param paths: list of paths to the images
    @param file_names: list of names of the images
    @param labelFile: path to the label file
    @return: the input, target and name lists
    """
    x = []
    y = []
    names = []
    characters = []
    for i in zip(paths, file_names):
        # Get inputs as images
        img = cv2.imread(i[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1500, 80))
        x.append(img)

        # Get target sentences
        label = get_y_from_file(i[1], labelFile)
        y.append(label)
        
        for ch in label:
            if ch not in characters:
                characters.append(ch)
        # 
        names.append(i[1])

    return x, y, names, characters
# Training script to train the Recurrent_CNN model for task 2. outputs the model to a file as well as results
if __name__ == "__main__":
    print("Training a Network for IAM Handwriting Database")

    img_path = "./Task_3/img/"
    img_names = os.listdir(img_path)
    paths = [img_path + name for name in img_names]

    label_gt = "./Task_3/iam_lines_gt.txt"
    with open(label_gt, "r") as f:
        label_gt = f.readlines()
    label_gt = [label.strip() for label in label_gt if label.strip()]

    input_images, target_strings, file_names, all_chars = load_data(paths, img_names, label_gt)

    # Models and Parameter

    print(f"Length of input images: {len(input_images)}")
    print(f"Length of target strings: {len(target_strings)}")
    print(f"Length of file names: {len(file_names)}")
    print(f"Total number of characters: {len(all_chars)}") 
    print(f"Characters: {all_chars}")
    print("\n")
    

    