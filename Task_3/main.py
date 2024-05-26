import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn

from network import *




def run_IAM(datapath : str) -> None:
    """
    Loads the best saved model and runs it on new IAM data available in filepath. 
    Expects path to be filled with images png type 
    Will print results and save prediction in a file
    @param datapath: path to the data to run the model on
    """

    return

if __name__ == "__main__":
    data_path = "./test" # INSERT PATH TO DATA
    print(f"Testing on New Data in path {data_path}")

    if os.path.exists(data_path):
        run_IAM(data_path)
    else:
        print("Data path does not exist")
    
