"""
Matisse_Segmentation_Approach

This script processes an image to segment it into lines of character bounding boxes, resizes each character image to a fixed size (38x48 by default), and pads the resized images to maintain their aspect ratio.

Usage:
    You can import this script as a module and use the `extract_and_resize_characters` function
    to process an image and return the resized character images.

Example:
    from Matisse_Segmentation_Approach import extract_and_resize_characters

    resized_chars = extract_and_resize_characters("path/to/your/image.jpg")

    for i, char_img in enumerate(resized_chars):
        cv2.imshow(f'Char {i}', char_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""

import cv2
import numpy as np
from utils import *


def find_lines(char_boxes, img_height):
    """
    Organizes character bounding boxes into lines based on their vertical positions.

    This function sorts character bounding boxes from right to left, then groups them into lines
    based on their vertical overlap. Each line is represented as a list of bounding boxes.
    Lines with fewer than 3 characters are discarded. The remaining lines are sorted
    from top to bottom based on their vertical position.

    Parameters:
    char_boxes (list of tuples): List of bounding boxes for characters. Each bounding box is represented
                                 as a tuple (x, y, w, h) where (x, y) is the top-left corner, and
                                 (w, h) are the width and height.
    img_height (int): The height of the image from which the characters were extracted.

    Returns:
    list of list of tuples: A list of lines, where each line is a list of bounding boxes (tuples).
                            The lines are sorted from top to bottom.
    """
    # Sort character boxes by the x coordinate in descending order for RTL languages
    char_boxes = sorted(char_boxes, key=lambda x: x[0], reverse=True)

    lines = []
    vertical_extension = img_height // 39  # Set vertical extension as 1/40th of the image height

    while char_boxes:
        # Start with the rightmost box (first in the list)
        rightmost_box = char_boxes[0]
        current_line = []

        # Establish a horizontal line across the y-midpoint of the rightmost box
        y_mid = rightmost_box[1] + rightmost_box[3] // 2
        line_min = y_mid - vertical_extension  # Extend vertically by 1/40th of image height
        line_max = y_mid + vertical_extension

        # Check all remaining boxes to see if they intersect with this line
        remaining_boxes = []
        for box in char_boxes:
            box_mid = box[1] + box[3] // 2
            if line_min <= box_mid <= line_max:
                current_line.append(box)
            else:
                remaining_boxes.append(box)

        # Add the current line to lines and use the remaining boxes for the next iteration
        lines.append(current_line)
        char_boxes = remaining_boxes

    # remove lines with less than 3 characters
    lines = [line for line in lines if len(line) >= 3]

    lines = sorted(lines, key=lambda line: line[0][1] if line else float('inf'))

    print("Number of lines found:", len(lines))

    return lines


def segment_Image(image_path):
    """
    Segments an image into lines of character bounding boxes.

    This function reads an image, applies adaptive thresholding to binarize it, and then finds
    contours representing character bounding boxes. It filters these bounding boxes based on
    height and width constraints, draws rectangles around them for visual debugging, and groups
    them into lines based on their vertical positions using the find_lines function.

    Parameters:
    image_path (str): The path to the image file to be processed.

    Returns:
    list of list of tuples: A list of lines, where each line is a list of bounding boxes (tuples).
                            The lines are sorted from top to bottom.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the height of the image
    img_height = img.shape[0]

    # Apply adaptive threshold to ensure the image is binary
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 30 < h < 200 and 10 < w < 100:
            char_img = img[y:y + h, x:x + w]
            char_boxes.append((x, y, w, h))

            char_images.append(char_img)
            # Draw rectangle for visual debugging
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Find lines
    lines = find_lines(char_boxes, img_height)

    # Show the image with detected characters and horizontal lines marked
    # cv2.imshow('Segmented Characters with Lines', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lines

def extract_and_resize_characters(img_path):
    """
    Extracts characters from an image and resizes them to a fixed size.

    This function processes an image to extract character bounding boxes, organizes them into lines,
    and resizes each character image to a fixed size (38x48 by default) with padding to maintain aspect ratio.

    Parameters:
    img_path (str): The path to the image file to be processed.

    Returns:
    list of numpy.ndarray: A list of resized character images.
    """
    lines = segment_Image(img_path)  # Segment the image into lines of characters
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale

    char_images_resized = []
    for line in lines:
        for box in line:
            x, y, w, h = box
            char_img = img[y:y + h, x:x + w]  # Extract the character image from the bounding box
            resized_char_img = resize_and_pad(char_img)  # Resize and pad the character image
            char_images_resized.append(resized_char_img)  # Add the resized image to the list

    return char_images_resized

def main():
    """
    Main function to demonstrate the usage of the character extraction and resizing.

    This function processes a given image to extract and resize character images, then displays them one by one.
    """
    path = "image-data/P123-Fg001-R-C01-R01-binarized.jpg"
    char_images_resized = extract_and_resize_characters(path)
    print("Number of characters found:", len(char_images_resized))

    # For debugging, let's visualize the resized character images
    for i, char_img in enumerate(char_images_resized):
        cv2.imshow(f'Char {i}', char_img)
        cv2.waitKey(0)
        # destroy tthis window
        cv2.destroyWindow(f'Char {i}')

if __name__ == "__main__":
    main()
