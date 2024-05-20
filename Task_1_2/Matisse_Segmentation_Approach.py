import cv2
import numpy as np



def segment_Image(image_path):
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



    # show morphed image for debugging
    # cv2.imshow('char', img_dilated)
    # cv2.waitKey(0)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_char_width = 110
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
    segment_Image(path)

if __name__ == "__main__":
    main()
