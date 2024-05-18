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
    # Apply thresholding    
    _, image = cv.threshold(image, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
    return image

def get_text_images():
    pass


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
#
##############################################

if __name__ == '__main__':
    print("Running the segmentation script only")

    testing_folder = "./Task_1_2/image-data/"   # Folder containing the images of the dead sea scrolls.
    images = get_image_paths(testing_folder, mode = USE_BINARY)

    segmented_images = []
    for image in images:
        full_image = cv.imread(testing_folder + image)
        full_image = cv.resize(full_image, (TEXT_HEIGHT, TEXT_WIDTH))

        # IF we are using the already binarized images, we do not need to convert them to binary again.
        if not USE_BINARY:
            full_image = convert_img_toBinary(full_image)

        # Visualize the full image
        cv.imshow("Full image", full_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # We segment the image.
        segmented = segment_Image(full_image)
        # Visualize the segmented images

        segmented_images.append(segmented)

    print("Segmentation complete.")
    print("The segmented images are stored in the segmented_images list.")
    # Could now continue with the classification of the images.
