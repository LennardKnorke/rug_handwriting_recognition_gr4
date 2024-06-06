import os
import cv2
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad, CHAR_SET, CHAR_TO_IDX, IDX_TO_CHAR, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_SEQ_LENGTH

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
        print("Number of unique characters: ", len(CHAR_SET))
        print("Characters: ", CHAR_SET)
        return

    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, idx):
        """
        @return: image (as image file path), (encoded target, target_length, target string)
        """
        # Read and preprocess image
        img = self.images_paths[idx]
        #img = cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE)
        #img = resize_and_pad(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        #img = img / 255.0
        #img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Convert target to encoded 128 long tensor
        target_enc = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.long) # IMPORTANT. DOUBLE CHECK IF PADDING IS SPACE BARS (1) OR BLANK (0)
        for i, char in enumerate(self.labels[idx]):
            target_enc[i] = CHAR_TO_IDX[char]
        target_length = torch.tensor(len(self.labels[idx]), dtype = torch.long)

        # Return the preprocessed image, encoded target, the strings length and the original string
        return img, (target_enc, target_length, img)