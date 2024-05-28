import cv2
import numpy as np


def find_lines(char_boxes, img_height):
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

    print("Number of lines found:", len(lines))

    return lines


def segment_Image(image_path):
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

    # Draw horizontal lines on the image for debugging
    line_thickness = max(1, img_height // 39)
    for line in lines:
        if line:
            y_mid = line[0][1] + line[0][3] // 2
            # Find the minimum and maximum x coordinates in the line
            min_x = min([box[0] for box in line])
            max_x = max([box[0] + box[2] for box in line])
            cv2.line(img, (min_x, y_mid), (max_x, y_mid), (255, 0, 255), line_thickness)

    # Show the image with detected characters and horizontal lines marked
    cv2.imshow('Segmented Characters with Lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return char_images


def main():
    path = "image-data/P123-Fg001-R-C01-R01-binarized.jpg"
    char_images = segment_Image(path)
    print("Number of characters found:", len(char_images))


if __name__ == "__main__":
    main()
