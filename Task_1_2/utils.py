import os
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump


TEXT_HEIGHT : int = 1000      # The height of the full unsegmented text
TEXT_WIDTH : int = 1000       # The width of the full unsegmented text

CHARACTER_HEIGHT : int = 48  # The height of a single character
CHARACTER_WIDTH : int = 38   # The width of a single character

USE_BINARY : bool = True       # Set to True if the images to use are already binarized. False if they are RGB and we binarize ourselves
BINARY_THRESHOLD : int = 100   # The threshold to use when binarizing the images.


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
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def file_to_img(path):
    img = cv2.imread(path,-1)
    img = cv2.resize(img, (38, 48))
    return img

def load_segmented_data(test_files=None, test=False, print_mean_dims=False):
    n = 5536
    if test_files is not None:
        if test: n = len(test_files)
        else: n -= len(test_files)
    subfolders = [f.path for f in os.scandir("img//segmented") if f.is_dir()]
    imgs = np.empty((n, 1, 48, 38))
    labels = np.empty(n, dtype='|S12')
    img_files = np.empty(n, dtype='|S500')
    if print_mean_dims: heights, widths = list(), list()

    i = 0
    for sf in subfolders:
        for img_file in os.listdir(sf):
            if img_file == "Kaf-final-00010-mo=erod-smo=13-shearangle=-8.23-decoco=1-artif.pgm": continue
            if test_files is not None and test != (img_file.encode() in test_files): continue
            img = file_to_img(os.path.join(sf, img_file))
            imgs[i] = img
            img_files[i] = img_file
            labels[i] = sf.split("\\")[-1]
            if print_mean_dims: heights.append(img.shape[0]); widths.append(img.shape[1])
            i += 1
    if print_mean_dims: print(f"Mean height {np.mean(heights)}, Std height {np.std(heights)}, Mean width {np.mean(widths)}, Std width {np.std(widths)}")
    return imgs, labels, img_files

def prepare_train_test_data():
    _, y, f = load_segmented_data()
    _, f_test, _, _ = train_test_split(f, y, test_size=0.1, stratify=y, random_state=42)
    np.save("test_files.npy", f_test)
    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit(y[:, np.newaxis])
    dump(encoder, "encoder.joblib")

def plot_loss(
    loss: float,
    line: Line2D,
    ep: int
):      
    plt.gca().set_ylim([0, loss*10])
    if ep == 0:
        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    plt.axvline(ep, alpha=0.0)
    line.set_xdata(np.append(line.get_xdata(), ep))
    line.set_ydata(np.append(line.get_ydata(), loss))

    plt.pause(0.00001)