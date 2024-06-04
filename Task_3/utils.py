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
        # if not pad or img.shape[0] > CHARACTER_HEIGHT or img.shape[1] > CHARACTER_WIDTH:
        #     img = cv2.resize(img, (CHARACTER_WIDTH, CHARACTER_HEIGHT))
        # else:
        #     h = (CHARACTER_HEIGHT - img.shape[0]) / 2
        #     w = (CHARACTER_WIDTH - img.shape[1]) / 2
        #     img = cv2.copyMakeBorder(img, math.floor(h), math.ceil(h),  math.floor(w), math.ceil(w), cv2.BORDER_CONSTANT)
        img = resize_and_pad(img)
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


def create_test_split(split):
    """
    Create and save a train-test split of the pre-segmented images in the Dead sea scrolls dataset.
    Only the names of the test images have to be saved.
    """
    _, labels, filenames = load_segmented_data()
    _, filenames_test, _, _ = train_test_split(filenames, labels, test_size=split/100, stratify=labels, random_state=42)
    np.save(get_test_split_filename(split), filenames_test)
    if not os.path.exists("encoder.joblib"): create_encoder()

def create_encoder():
    """
    Create and save an encoder.
    """
    _, labels, _ = load_segmented_data()
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit(labels[:, np.newaxis])
    dump(encoder, "encoder.joblib")

def get_test_split_filename(split):
    return f"test_split_{split}.npy"

def plot_interactive(
    aug_loss: float,
    rec_loss: float,
    acc: float,
    aug_loss_line: Line2D,
    rec_loss_line: Line2D,
    acc_line: Line2D,
    lta=True,
    train_recog=True
):
    """
    Add new data to the interactive plots with augmentation loss, recognizer (classifier) loss, and accuracy.
    """
    if lta: ep = len(aug_loss_line.get_ydata())
    else: ep = len(rec_loss_line.get_ydata())
    
    to_update = []
    if lta:
        to_update = ["aug"]
        if ep == 0: plt.figure("aug"); plt.xlabel('Batch'); plt.ylabel('Augmentation Loss'); plt.gca().yaxis.tick_right()
    if train_recog:
        to_update.extend(["rec", "acc"])
        if ep == 0:
            plt.figure("rec"); plt.xlabel('Batch'); plt.ylabel('Classifier Loss'); plt.gca().yaxis.tick_right()
            plt.figure("acc"); plt.xlabel('Batch'); plt.ylabel('Accuracy'); plt.gca().yaxis.tick_right()

    for i, x in enumerate(["aug", "rec", "acc"]):
        if x not in to_update: continue
        plt.figure(x)
        ax = plt.gca()
        line = [aug_loss_line, rec_loss_line, acc_line][i]

        data = line.get_ydata()
        if x == "aug": ax.set_ylim([0, sum(data[-20:])/10])
        if x == "rec": ax.set_ylim([0, sum(data[-20:])/10])
        if x == "acc": ax.set_ylim([1 - (1 - (sum(data[-20:])/20))*2, 1])

        plt.axvline(ep, alpha=0.0)
        line.set_xdata(np.append(line.get_xdata(), ep))
        line.set_ydata(np.append(line.get_ydata(), [aug_loss, rec_loss, acc][i]))

    plt.pause(0.00001)

def resize_and_pad(image, size=(CHARACTER_WIDTH, CHARACTER_HEIGHT)):
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

    color = [255, 255, 255]  # Padding color (white
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Add padding

    return new_image

def uniquify(path, find=False):
    """
    Adapt the filename so that it doesn't overwrite.
    @param path: Path to possible modify.
    @param find: Instead of modifying path, find the name of the latest version.
    @return (Possibly) modified pathname such as 'file.ext' 'file (1).ext' or 'file (2).ext'
            If find is True, then it returns the name of the latest version.
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    if find:
        if counter == 2: return filename + extension
        elif counter > 2: return filename + " (" + str(counter-2) + ")" + extension
    return path

def hebrewize(roman):
    d = {
         'Alef': 'א',
         'Ayin': 'ע',
         'Bet': 'ב',
         'Dalet': 'ד',
         'Gimel': 'ג',
         'He': 'ה',
         'Het': 'ח',
         'Kaf': 'כ',
         'Kaf-final': 'ך',
         'Lamed': 'ל',
         'Mem': 'ם',
         'Mem-medial': 'מ',
         'Nun-final': 'ן',
         'Nun-medial': 'נ',
         'Pe': 'פ',
         'Pe-final': 'ף',
         'Qof': 'ק',
         'Resh': 'ר',
         'Samekh': 'ס',
         'Shin': 'ש',
         'Taw': 'ת',
         'Tet': 'ט',
         'Tsadi-final': 'ץ',
         'Tsadi-medial': 'צ',
         'Waw': 'ו',
         'Yod': 'י',
         'Zayin': 'ז'
    }
    return d[roman]