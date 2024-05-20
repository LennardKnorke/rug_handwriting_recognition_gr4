import numpy as np


from segmentation import segment_Image, convert_img_toBinary
from augmentation import aug_temp
from classification import CNN_Network, save_hyperparameters








##############################################
# Main script. This script will run the entire pipeline, given the folder of input images.
##############################################


if __name__ == '__main__':
    print("RUNNING TASK 1. THE DEAD SEA SCROLLS")
    # Folder containing the testing image of the first task.
    folder_path_part1 = "./Task_1_2/image-data/"    # CHANGE WHEN RUNNING WITH NEW IMAGES



    print("RUNNING TASK 2. THE IAM DATASET")
    # Folder containing the images of the IAM dataset.
    image_folder = "./Task_3/"                       # CHANGE WHEN RUNNING WITH NEW IMAGES

    pass