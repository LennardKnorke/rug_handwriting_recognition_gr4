# Import third party modules
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Tuple

# Our modules
from network import *
SEED = 42

# Learning Makros
BATCH_SIZE = 32
N_EPOCHS = 240
N_FOLDS = 10
TEST_RATIO = 0.2
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_y_from_file(name : str, labelFile) -> str:
    """
    Search for a name in list and return line below it.
    @param name: name to search for
    @param labelFile: list of labels
    @return: the label of the name
    """
    for i, row in enumerate(labelFile):
        if name == row:
            return labelFile[i + 1]
    return ""


class IAM_Dataset(Dataset):
    def __init__(self, images_folder : str, labels_file):
        """
        Complete dataset to be used for training and testing
        @param images_folder: folder containing the images
        @param labels_file: file containing the labels
        """
        # Set up Image folder
        self.images_folder = images_folder
        self.images_files = os.listdir(images_folder)
        self.images_paths = [images_folder + name for name in self.images_files]

        # Read gt file with labels
        with open(labels_file, "r") as f:
            label_gt = f.readlines()
        label_gt = [label.strip() for label in label_gt if label.strip()]

        # Get target strings and all characters used
        self.labels = []
        self.chars = []
        self.n_chars = 0
        for file_name in self.images_files:
            # Get target label (sentence)
            label = get_y_from_file(file_name, label_gt)
            assert label != "", "No label found for image " + file_name
            label = " " + label + " "
            self.labels.append(label)

            # Get all characters that are used
            for ch in label:
                if ch not in self.chars:
                    self.chars.append(ch)
            # Update Max Length
            if len(label) > self.n_chars:
                self.n_chars = len(label)

        self.char_index = dict(zip(self.chars, range(len(self.chars))))
        
        print("Number of images: ", len(self.images_files))
        print("Max length of target string: ", self.n_chars)
        print("Number of unique characters: ", len(self.chars))
        print("Characters: ", self.chars)
        return

    def strings_onehot(self, string: str) -> torch.Tensor:
        targets = torch.zeros((
            self.n_chars,
            len(self.chars),
        ))

        for ch_idx, char in enumerate(string):
            targets[ch_idx, self.char_index[char]] = 1.0
        return targets

    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img, self.strings_onehot(self.labels[idx])
    
def strings_to_targets(strings : Tuple[str]) -> torch.Tensor:
    """
    Convert the a tuple with the full target strings into a tensor
    @param strings: tuple of strings
    @return: tensor with the targets
    """
    # Create a list of lists with each character as a target
    targets = [[char for char in full_string] for full_string in strings]
    # Flatten
    targets_flat = [char for sublist in targets for char in sublist]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(targets_flat)
    targets_enc = [encoder.transform(y) for y in targets]
    #targets_enc = np.array(targets) + 1

    print(np.unique(targets_enc))
    print(targets_enc)
    return targets_enc


def train_model(model : nn.Module, 
                training_data : Dataset,
                training_loader : DataLoader,
                testing_data : Dataset,
                testing_loader : DataLoader,
                optimizer : optim,
                scheduler : optim.lr_scheduler
                )-> list:
    """
    Train a model on a dataset and run validation with given optimization/scheduler
    @param model: model to train
    @param training_data: dataset to train on
    @param training_loader: loader for the training data
    @param validation_data: dataset to validate on
    @param validation_loader: loader for the validation data
    @param optimizer: optimizer to use for training
    @param scheduler: scheduler to use for training
    @return: history of the training and validation loss/accurracy as a list
    """
    history = []


    for ep in range(N_EPOCHS):
        loss_train = 0.0
        acc_train = 0.0

        # Training Loop
        for images, full_label_strings in training_loader:
            #targets = strings_to_targets(full_label_strings)
            targets = full_label_strings
            optimizer.zero_grad()
            model.train()

            images = images.to(DEVICE)

            # Forward Pass
            pred = model(images)
            print(pred)

            # Backward Pass

        
        # Testing Loop
        loss_test = 0.0
        acc_test = 0.0
        with torch.no_grad():
            for images, labels in testing_loader:
                model.eval()

                images = images.to(DEVICE)
                print(images.shape, labels)
                # Forward Pass
                pred = model(images)

        # End of Epoch
        history.append((
            ep,
            (loss_train, acc_train),
            (loss_test, acc_test)
        ))
    return history

# Training script to train the Recurrent_CNN model for task 2. outputs the model to a file as well as results
if __name__ == "__main__":
    print("Training a Network for IAM Handwriting Database")


    # Prepapre data (training and testing data)
    Data = IAM_Dataset(images_folder = "./Task_3/img/", 
                       labels_file = "./Task_3/iam_lines_gt.txt"
                       )
    train_data, test_data = train_test_split(Data, test_size = TEST_RATIO, random_state = SEED)
    print("Training data: ", len(train_data))
    print("Testing data: ", len(test_data))


    # Set up KFold
    kFolds = KFold(n_splits = N_FOLDS, shuffle = True, random_state = SEED)
    # Parameters to search for
    parameters = {
        "learning_rate": [0.001, 0.0001],
        "beta": [(0.9, 0.999), (0.95, 0.999)],
        "rnn_size": [128, 256],
        "rnn_dropout": [0.25, 0.5]
    }
    paramList = list(itertools.product(*parameters.values()))

    best_loss = 1000000
    best_params = []

    # Train different models for multiple folds
    for params in paramList:
        print("Training with parameters: ", params)
        learning_rate, beta, rnn_size, rnn_dropout = params

        losses = []
        for train_idx, test_idx in kFolds.split(train_data):
            # Set up CRNN model + optimizer
            model = Recurrent_CNN(Data.n_chars, 
                                  rnn_size_1=rnn_size, rnn_dropout_1=rnn_dropout,
                                  rnn_size_2=rnn_size, rnn_dropout_2=rnn_dropout,
                                  rnn_size_3=rnn_size, rnn_dropout_3=rnn_dropout
                                  )
            model.to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=beta)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

            # Set up data for the current fold
            train_data_fold = Subset(train_data, train_idx)
            test_data_fold = Subset(train_data, test_idx)
            train_loader = DataLoader(train_data_fold, batch_size = BATCH_SIZE, shuffle = True)
            test_loader = DataLoader(test_data_fold, batch_size = BATCH_SIZE, shuffle = False)

            #
            history = train_model(model = model,
                                  training_data = train_data_fold,
                                  training_loader = train_loader,
                                  testing_data = test_data_fold,
                                  testing_loader = test_loader,
                                  optimizer = optimizer,
                                  scheduler = scheduler)
            
            # Remember last val loss of current fold
            losses.append(history[-1][1][0])
        # After all folds, check the mean loss and remember best params
        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params
            print("New best model found with parameters: ", params)
    
    ##########################################################################################################
    # Train the best model with all the training data and test on testing data
    # Save the performance and model at the end
    ##########################################################################################################
    print("Best model found with parameters: ", best_params)
    learning_rate, beta, rnn_size, rnn_dropout = best_params
    model = Recurrent_CNN(Data.n_chars, 
                          rnn_size_1=rnn_size, rnn_dropout_1=rnn_dropout,
                          rnn_size_2=rnn_size, rnn_dropout_2=rnn_dropout,
                          rnn_size_3=rnn_size, rnn_dropout_3=rnn_dropout)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=beta)

    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    history = train_model(model = model,
                          training_data = train_data,
                          training_loader = train_loader,
                          validation_data = test_data,
                          validation_loader = test_loader,
                          optimizer = optimizer,
                          scheduler = scheduler
                          )
    
    # Save the model
    torch.save(model.state_dict(), "./Task_3/IAM_the_best_model.pth")
    
    # Plot performance
    plt.figure()
    plt.plot(history[0][0], label="Training Loss")
    plt.plot(history[1][0], label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("./Task_3/loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(history[1][1], label="Testing Accuracy")
    plt.plot(history[0][1], label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("./Task_3/accuracy_plot.png")
    plt.close()
    # Done
