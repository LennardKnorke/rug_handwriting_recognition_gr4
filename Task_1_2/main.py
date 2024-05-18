import numpy as np


from segmentation import segment_Image, convert_img_toBinary
from augmentation import aug_temp
from classification import CNN_Network, save_hyperparameters








##############################################
# Main script. This script will run the entire pipeline, given the folder of input images.
##############################################


if __name__ == '__main__':
    print("This is the main script")


    image_folder = "./Task_1_2/"   # Folder containing the images of the dead sea scrolls. 
                                        # CHANGE FOR TESTING

    pass