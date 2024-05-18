import numpy as np
import cv2 as cv
import os
from utils import get_image_paths
##############################################
# CLASSES AND FUNCTIONS FOR THE SEGMENTATION SECTION
##############################################

def convert_img_toBinary(image):
    pass


def segment_Image(fullImage, mode : str = "binary") -> list:
    """
    This function segments the image into different regions which contain the characters of the dead sea scrolls.
    @param fullImage: The full image of the dead sea scrolls.
    @param mode: state of the image. binary, RGB, greyscale. Default is binary.
    @return list of images, each image containing a single character. In order of appearance in the original image.
    """
    
    single_images = []
    return single_images



##############################################
#
##############################################
USE_BINARY_ONLY = True
if __name__ == '__main__':
    print("Running the segmentation script only")

    testing_folder = "./Task_1_2/image-data/"   # Folder containing the images of the dead sea scrolls.
    images = get_image_paths(testing_folder, mode = USE_BINARY_ONLY)

    segmented_images = []
    for image in images:
        full_image = cv.imread(testing_folder + image)
        segmented_images.append(segment_Image(full_image, mode = USE_BINARY_ONLY))

    print("Segmentation complete.")
    print("The segmented images are stored in the segmented_images list.")
    # Could now continue with the classification of the images.
