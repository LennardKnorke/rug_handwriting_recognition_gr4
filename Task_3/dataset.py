import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128

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

        # Get target strings and all characters used
        
        self.chars = (' ',
                      'a','A','b','B','c','C','d','D','e','E','f','F','g','G','h','H','i','I','j','J','k','K','l','L','m','M','n','N','o','O','p','P','q','Q','r','R','s','S','t','T','u','U','v','V','w','W','x','X','y','Y','z','Z',
                      '0','1','2','3','4','5','6','7','8','9',
                      '\\','\'','!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','=','>','?','_')
        self.n_chars = len(self.chars)
        self.chars_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_chars = {idx + 1: char for idx, char in enumerate(self.chars)}

        self.labels = []
        for file_name in self.images_files:
            # Get target label (sentence)
            label = get_y_from_file(file_name, label_gt)
            assert label != "", "No label found for image " + file_name
            label = " " + label + " "
            self.labels.append(label)
            
        
        print("Number of images: ", len(self.images_files))
        print("Max length of target string: ", self.n_chars)
        print("Number of unique characters: ", len(self.chars))
        print("Characters: ", self.chars)
        return

    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img / 255.0
        # Resize and Padd image
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        target = torch.ones(128, dtype=torch.long)
        for i, char in enumerate(self.labels[idx]):
            target[i] = self.chars_to_idx[char]
        target_length = torch.tensor(len(self.labels[idx]), dtype = torch.long)
        """
        # Set up target tensor one hot encoded
        target = torch.zeros((128, self.n_chars) , dtype=torch.float32)
        for i, char in enumerate(self.labels[idx]):
            target[i, self.chars_to_idx[char]] = 1.0
        # set beyond label length idx 0 to 1
        target[i+1:, 0] = 1.0
        """

        return img, (target, target_length)
    
    def resizeImg(self, img, width = None, height = None):

        if height is None and width is not None:
            scale = float(width) / img.shape[1]
            height = int(img.shape[0] * scale)
        
        if width is None and height is not None:
            scale = float(height) / img.shape[0]
            width = int(img.shape[1] * scale)
        
        return cv2.resize(img, (width, height)).astype(np.float32)