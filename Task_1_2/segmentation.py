import numpy as np
import cv2 as cv
from utils import *
##############################################
# CLASSES AND FUNCTIONS FOR THE SEGMENTATION SECTION
##############################################

def convert_img_toBinary(image : np.ndarray) -> np.ndarray:
    """
    This function converts an image to binary.
    @param image: The image to convert.
    @return The binarized image.
    """
    # Convert to greyscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # SIMPLE thresholding    
    # _, image = cv.threshold(image, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)

    # Alternative, gaussian thresholding
    _, image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return image

def get_text_images(image_list : list) -> list:
    images = []

    for image_path in image_list:
        print("Reading image", image_path)
        img = cv.imread(image_path)
        resized_img = cv.resize(img, (TEXT_HEIGHT, TEXT_WIDTH))
        binarized_img = convert_img_toBinary(resized_img) if not USE_BINARY else resized_img
        images.append(binarized_img)
    return images

def segment_Image(fullImage) -> list:
    """
    This function segments the image into different regions which contain the characters of the dead sea scrolls.
    @param fullImage: The full image of the dead sea scrolls.
    @param mode: state of the image. binary, RGB, greyscale. Default is binary.
    @return list of images, each image containing a single character. In order of appearance in the original image.
    """
    
    single_images = []
    return single_images



##############################################
# Testing script. Not used in the final implementation.
##############################################
if __name__ == '__main__':
    print("Running the segmentation script only")
    testing_folder = "./Task_1_2/image-data/"   # Folder containing the images of the dead sea scrolls.
    image_list = get_image_paths(testing_folder)
    images = get_text_images(image_list)

    segmented_images = []
    for image in images:
        segmented_images.append(segment_Image(image))

    print("Segmentation complete.")
    print("The segmented images are stored in the segmented_images list.")
    # Could now continue with the classification of the images.
