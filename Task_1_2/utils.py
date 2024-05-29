import os
import cv2
import numpy as np
import torch.nn as nn
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import math


TEXT_HEIGHT : int = 1000      # The height of the full unsegmented text (Optional)
TEXT_WIDTH : int = 1000       # The width of the full unsegmented text (Optional)

CHARACTER_HEIGHT : int = 70  # The height of a single character for Dead sea scrolls
CHARACTER_WIDTH : int = 50   # The width of a single character for Dead sea scrolls

USE_BINARY : bool = True       # Set to True if the images to use are already binarized. False if they are RGB and we binarize ourselves
BINARY_THRESHOLD : int = 100   # The threshold to use when binarizing the images.

N_EPOCHS : int = 20
BATCH_SIZE : int = 64

RECOGNIZER_LOSS_FUNCTION = nn.CrossEntropyLoss()


def get_image_paths(folder_path : str) -> list:
    """
    This function reads all the images in a folder and returns them as a list.
    @param folder_path: The path to the folder containing the images.
    @return list of images.
    """
    image_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            file_type = file.split("-")[-1].split(".")[0]
            if (USE_BINARY and file_type == "binarized") or (not USE_BINARY and file_type == "R01"):
                image_paths.append(folder_path+file)
    print("Found", len(image_paths), "images.")
    return image_paths

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

def file_to_img(path : str, resize=True, pad=True) -> np.ndarray:
    """
    Read an image file and return it resized as a numpy array.
    @param path: The path to the image file.
    @param resize: Whether the image should be resized to CHARACTER_HEIGHT x CHARACTER_WIDTH.
    @param pad: Whether image smaller than CHARACTER_HEIGHT x CHARACTER_WIDTH should be padded instead.
    @return The image as a numpy array.
    """
    img = cv2.imread(path,-1)
    if resize:
        if not pad or img.shape[0] > CHARACTER_HEIGHT or img.shape[1] > CHARACTER_WIDTH:
            img = cv2.resize(img, (CHARACTER_WIDTH, CHARACTER_HEIGHT))
        else:
            h = (CHARACTER_HEIGHT - img.shape[0]) / 2
            w = (CHARACTER_WIDTH - img.shape[1]) / 2
            img = cv2.copyMakeBorder(img, math.floor(h), math.ceil(h),  math.floor(w), math.ceil(w), cv2.BORDER_CONSTANT)
    return img

def show_image_dimensions():
    """
    Plot the widths and heights of the pre-segmented images in the Dead sea scrolls dataset
    """
    subfolders = [f.path for f in os.scandir("img//segmented") if f.is_dir()]
    heights, widths = list(), list()
    for sf in subfolders:
        for img_file in os.listdir(sf):
            if img_file == "Kaf-final-00010-mo=erod-smo=13-shearangle=-8.23-decoco=1-artif.pgm": continue
            img = file_to_img(os.path.join(sf, img_file), resize=False)
            heights.append(img.shape[0]); widths.append(img.shape[1])
    print(f"Mean height {np.mean(heights)}, Std height {np.std(heights)}, Mean width {np.mean(widths)}, Std width {np.std(widths)}")

    plt.figure(); plt.xlabel('n'); plt.ylabel('width'); plt.plot(np.sort(widths))
    plt.figure(); plt.xlabel('n'); plt.ylabel('height'); plt.plot(np.sort(heights))
    plt.show()


def load_segmented_data(test_files=None, test=False):
    """
    Load segmented Dead sea scrolls images as a numpy array.
    @param test_files: List of image filenames used for the test set.
    @param test: Whether the test images should be loaded. If False while test_files is given,
                    then the training images are loaded (the opposite of test_files).
    @return A numpy array of images as numpy arrays.
    """
    n = 5536
    if test_files is not None:
        if test: n = len(test_files)
        else: n -= len(test_files)
    subfolders = [f.path for f in os.scandir("img//segmented") if f.is_dir()]
    imgs = np.empty((n, 1, CHARACTER_HEIGHT, CHARACTER_WIDTH))
    labels = np.empty(n, dtype='|S12')
    img_files = np.empty(n, dtype='|S500')

    i = 0
    for sf in subfolders:
        for img_file in os.listdir(sf):
            if img_file == "Kaf-final-00010-mo=erod-smo=13-shearangle=-8.23-decoco=1-artif.pgm": continue
            if test_files is not None and test != (img_file.encode() in test_files): continue

            img = file_to_img(os.path.join(sf, img_file))
            imgs[i] = img
            img_files[i] = img_file
            labels[i] = sf.split("\\")[-1]

            i += 1
            
    return imgs, labels, img_files

def prepare_train_test_data():
    """
    Create and save a train-test split of the pre-segmented images in the Dead sea scrolls dataset.
    Only the names of the test images and the encoder have to be saved.
    """
    _, y, f = load_segmented_data()
    _, f_test, _, _ = train_test_split(f, y, test_size=0.1, stratify=y, random_state=42)
    np.save("test_files.npy", f_test)
    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit(y[:, np.newaxis])
    dump(encoder, "encoder.joblib")

def plot_interactive(
    aug_loss: float,
    rec_loss: float,
    acc: float,
    aug_loss_line: Line2D,
    rec_loss_line: Line2D,
    acc_line: Line2D
):
    """
    Add new data to the interactive plots with augmentation loss, recognizer (classifier) loss, and accuracy.
    """
    ep = len(aug_loss_line.get_ydata())
    if ep == 0:
        plt.figure("aug"); plt.xlabel('Batch'); plt.ylabel('Augmentation Loss')
        plt.figure("rec"); plt.xlabel('Batch'); plt.ylabel('Classifier Loss')
        plt.figure("acc"); plt.xlabel('Batch'); plt.ylabel('Accuracy'); plt.gca().set_ylim([0, 1])
        
    for i, x in enumerate(["aug", "rec", "acc"]):
        plt.figure(x)
        line = [aug_loss_line, rec_loss_line, acc_line][i]

        if x != "acc":
            losses = line.get_ydata()
            if x == "rec": plt.gca().set_ylim([0, sum(losses[-20:])/2])
            if x == "aug": plt.gca().set_ylim([0, sum(losses[-20:])/10])

        plt.axvline(ep, alpha=0.0)
        line.set_xdata(np.append(line.get_xdata(), ep))
        line.set_ydata(np.append(line.get_ydata(), [aug_loss, rec_loss, acc][i]))

    plt.pause(0.00001)