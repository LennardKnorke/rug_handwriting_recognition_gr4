import numpy as np


from segmentation import segment_Image, convert_img_toBinary
from augmentation import aug_temp
from classification import CNN_Network, save_hyperparameters

##############################################
# Main script. This script will run the entire pipeline, given the folder of input images.
##############################################

def run():
    # 1. Read and transform data
    # 2. Segment
    # 3. Classify
    # 4. Produce output

    pass


if __name__ == '__main__':
    print("This is the main script")

    image_folder = "./Task_1_2/"   # Folder containing the images of the dead sea scrolls. 
                                        # CHANGE FOR TESTING

    pass