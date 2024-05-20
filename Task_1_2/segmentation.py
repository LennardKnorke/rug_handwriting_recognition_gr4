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



def segment_Image(image_path : str) -> list:
    """
    Segment an image into individual characters. and displays the results. Return a list of the segmented characters.
    @param image_path: The path to the image to segment.
    @return A list of the segmented characters (numpy arrays representing the characters)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # show normal image for debugging
    # cv2.imshow('char', img)
    # cv2.waitKey(0)


    # Apply adaptive threshold to ensure the image is binary
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    # Apply morphological dilation to connect character parts
    # doesnt seem to work well
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_dilated = cv2.dilate(img, kernel, iterations=1)

    # check with erosion (maybe fixes the lines touching each other)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # Adjust kernel size as needed
    img_eroded = cv2.erode(img, erosion_kernel, iterations=1)


    # show eroded
    cv2.imshow('char', img_eroded)
    cv2.waitKey(0)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(img_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)


        if 30 < h < 200:
            char_img = img[y:y + h, x:x + w]

            # show char image for debugging
            # cv2.imshow('char', char_img)
            # cv2.waitKey(0)

            char_images.append(char_img)
            # Draw rectangle for visual debugging
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the image with detected characters marked
    cv2.imshow('Segmented Characters', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return char_images

def main():
    path = "image-data/P123-Fg001-R-C01-R01-binarized.jpg"
    char_images = segment_Image(path)
    print("Number of characters found:", len(char_images))

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
