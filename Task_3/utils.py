import cv2
import torch
import torchvision
import numpy as np
import fastwer
import imageio.v3 as iio
import matplotlib.pyplot as plt

# MAKROS
SEED : int = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_WIDTH : int = 1024 #Either 1024 or 128
IMAGE_HEIGHT = 128 #Either 128 or 32
MAX_SEQ_LENGTH = 128

CHAR_SET = (' ',
            'a','A','b','B','c','C','d','D','e','E','f','F','g','G','h','H','i','I','j','J','k','K','l','L','m','M','n','N','o','O','p','P','q','Q','r','R','s','S','t','T','u','U','v','V','w','W','x','X','y','Y','z','Z',
            '0','1','2','3','4','5','6','7','8','9',
            '\\','\'','!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','=','>','?','_')
N_CHARS = len(CHAR_SET)
CHAR_TO_IDX : dict = {char: idx + 1 for idx, char in enumerate(CHAR_SET)} #convert chars to ints, 0 is reserved for padding
IDX_TO_CHAR : dict = {idx + 1: char for idx, char in enumerate(CHAR_SET)} #convert ints to chars, 0 is reserved for padding


def preprocess_batch(images : list)->torch.Tensor:
    """
    Preprocess the batch of images for the network
    @param images: Images to process
    @return: batch of processed images
    """
    new_images = []
    for img in images:
        img = resize_and_pad(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        new_images.append(img)
    return torch.stack(new_images, dim=0)

def load_image_batch(image_paths):
    """
    Load the batch of images
    @param image_paths: paths for images
    @return: batch of unprocessed images (for before augmentation)
    """
    batch = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        batch.append(img)
    
    return batch

def resize_and_pad(image, size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Resizes and pads an image to a specified size while maintaining the aspect ratio.

    This function resizes the input image so that it fits within the specified size (38x48 by default),
    while maintaining its aspect ratio. It then pads the resized image with black pixels to match the
    specified size.

    Parameters:
    image (numpy.ndarray): The input image to be resized and padded.
    size (tuple): The desired output size (width, height). Default is (38, 48).

    Returns:
    numpy.ndarray: The resized and padded image.
    """
    h, w = image.shape
    scale = min(size[0] / w, size[1] / h)  # Calculate the scaling factor to maintain aspect ratio
    new_w = int(w * scale)  # Calculate new width
    new_h = int(h * scale)  # Calculate new height
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize the image

    delta_w = size[0] - new_w  # Calculate padding width
    delta_h = size[1] - new_h  # Calculate padding height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)  # Distribute padding height evenly
    left, right = delta_w // 2, delta_w - (delta_w // 2)  # Distribute padding width evenly

    color = [255, 255, 255]  # Padding color (white)
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Add padding

    return new_image


def ctc_decode(pred_ints : torch.Tensor, blank : int = 0) -> list:
    """
    Decode the batch output of the CTC module
    @param pred: output of the CRNN model (BatchSize, 128)
    @param blank: integer for spaces
    @return: decoded strings (BatchSize, Strings of variable length)
    """
    decoded_strs = []

    # Loop through all samples
    for i in range(pred_ints.shape[0]):
        string = ""
        prev_int = blank
        # Loop through all characters
        for j in range(pred_ints.shape[1]):
            # Append integers, skip blanks and duplicates
            index = pred_ints[i, j].item()
            if index != blank and index != prev_int:
                string += IDX_TO_CHAR[index]
                prev_int = index
        decoded_strs.append(string)

    return decoded_strs

def get_error_rates(predictions : list, targets : list) -> tuple:
    """
    Calculate the Word Error Rate and Character Error Rate between two lists of strings
    @param pred: list of predicted strings
    @param target: list of target strings
    @return: Word Error Rate, Character Error Rate
    """
    avg_wer : float = 0.0
    avg_cer : float = 0.0
    n = len(predictions)

    for x, y in zip(predictions, targets):
        avg_wer += fastwer.score_sent(x, y)
        avg_cer += fastwer.score_sent(x, y, char_level=True)
    avg_cer /= n
    avg_wer /= n

    return avg_wer, avg_cer

def levenstein_distance(str1 : str, str2 : str):
    """
    Calculate levenshtein distance:
    Taken from https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/!

    @param str1: predicted string
    @param str2: target string
    @return: levenshtein distance as int
    """
    # Get the lengths of the input strings
    m = len(str1)
    n = len(str2)
 
    # Initialize two rows for dynamic programming
    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)
 
    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = i
 
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Choose the minimum cost operation
                curr_row[j] = 1 + min(
                    curr_row[j - 1],  # Insert
                    prev_row[j],      # Remove
                    prev_row[j - 1]    # Replace
                )
 
        # Update the previous row with the current row
        prev_row = curr_row.copy()
 
    # The final element in the last row contains the Levenshtein distance
    return curr_row[n]

def create_gif(image_list, gif_name, duration=0.1):
    """
    Create a gif from a list of images.
    @param image_list: The list of images to use.
    @param gif_name: The name of the gif file to save.
    @param duration: The duration of each frame in the gif.
    """
    frames = []
    for image in image_list:
        frames.append(image)
    iio.imwrite(gif_name, frames, duration=duration, loop=0)
    return