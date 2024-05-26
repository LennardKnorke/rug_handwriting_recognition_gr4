import numpy as np
import os

from segmentation import *
from augmentation import *
from classification import *






def run_DSS(data_path : str) -> None:
    """
    Loads the best saved model and runs it on new DSS data available in filepath. 
    Expects path to be filled with images png type 
    Will print results and save prediction in a file
    @param datapath: path to the data to run the model on
    """

    return

##############################################
# Main script. This script will run the entire pipeline, given the folder of input images.
##############################################
if __name__ == '__main__':
    print("RUNNING TASK 1. THE DEAD SEA SCROLLS")
    # Folder containing the testing image of the first task.
    folder_path_part1 = "./folder"    # CHANGE WHEN RUNNING WITH NEW IMAGES

    if os.path.exists(folder_path_part1):
        run_DSS(folder_path_part1)
    else:
        print("Data path does not exist")
    