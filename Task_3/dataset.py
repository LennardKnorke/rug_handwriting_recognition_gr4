import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128

char_set = (' ',
            'a','A','b','B','c','C','d','D','e','E','f','F','g','G','h','H','i','I','j','J','k','K','l','L','m','M','n','N','o','O','p','P','q','Q','r','R','s','S','t','T','u','U','v','V','w','W','x','X','y','Y','z','Z',
            '0','1','2','3','4','5','6','7','8','9',
            '\\','\'','!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','=','>','?','_')
n_chars = len(char_set)
chars_to_idx = {char: idx + 1 for idx, char in enumerate(char_set)}
idx_to_chars = {idx + 1: char for idx, char in enumerate(char_set)}


def get_y_from_file(name : str, labelFile) -> str:
    """
    Search for a name in list and return line below it.
    @param name: name to search for
    @param labelFile: list of labels
    @return: the label of the name
    """
    for i, row in enumerate(labelFile):
        if name == row:
            return labelFile[i + 1]
    return ""


class IAM_Dataset(Dataset):
    def __init__(self, images_folder : str, labels_file):
        """
        Complete dataset to be used for training and testing
        @param images_folder: folder containing the images
        @param labels_file: file containing the labels
        """
        # Set up Image folder
        self.images_folder = images_folder
        self.images_files = os.listdir(images_folder)
        self.images_paths = [images_folder + name for name in self.images_files]

        # Read gt file with labels
        with open(labels_file, "r") as f:
            label_gt = f.readlines()
        label_gt = [label.strip() for label in label_gt if label.strip()]

        # Get target strings
        self.labels = []
        for file_name in self.images_files:
            label = get_y_from_file(file_name, label_gt)
            assert label != "", "No label found for image " + file_name
            label = " " + label + " " # Padding with space bars helps training
            self.labels.append(label)
            
        # Overview of the dataset and characters available
        print("Number of images: ", len(self.images_files))
        print("Max length of target string: ", max([len(label) for label in self.labels]))
        print("Number of unique characters: ", len(char_set))
        print("Characters: ", char_set)
        return

    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, idx):
        """
        @return: image, (onehot_targets, target_length)
        """
        # Read and preprocess image
        img = cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = resize_and_pad(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Convert target to encoded 128 long tensor
        target = torch.ones(128, dtype=torch.long) # IMPORTANT. DOUBLE CHECK IF PADDING IS SPACE BARS (1) OR BLANK (0)
        for i, char in enumerate(self.labels[idx]):
            target[i] = self.chars_to_idx[char]
        target_length = torch.tensor(len(self.labels[idx]), dtype = torch.long)

        return img, (target, target_length)