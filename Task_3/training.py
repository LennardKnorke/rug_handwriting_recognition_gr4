# Import third party modules
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchsummary import summary

# Our modules
from network import *
from dataset import *
SEED = 42

# Learning Makros
BATCH_SIZE = 32
N_EPOCHS = 100
N_FOLDS = 3
TEST_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model : nn.Module, 
                training_loader : DataLoader,
                testing_loader : DataLoader,
                optimizer : optim,
                scheduler : optim.lr_scheduler
                )-> list:
    """
    Train a model on a dataset and run validation with given optimization/scheduler
    @param model: model to train
    @param training_loader: loader for the training data
    @param validation_loader: loader for the validation data
    @param optimizer: optimizer to use for training
    @param scheduler: scheduler to use for training
    @return: history of the training and validation loss/accurracy as a list
    """
    history = []
    model.to(DEVICE)
    for ep in tqdm(range(N_EPOCHS)):
        loss_train = 0.0
        acc_train = 0.0
        loss_fn = nn.CTCLoss()

        # Training Loop
        for images, targets in training_loader:
            targets_output, targets_length = targets
            optimizer.zero_grad()
            model.train()

            images = images.to(DEVICE)

            # Forward Pass
            pred_rnn, pred_ctc = model(images)
            #shape = (BatchSize, SeqLen, NumClasses)
            # Convert to (SeqLen, BatchSize, NumClasses) for loss
            pred_rnn = pred_rnn.permute(1, 0, 2).log_softmax(2)
            pred_ctc = pred_ctc.permute(1, 0, 2).log_softmax(2)

            pred_int = pred_rnn.permute(1, 0, 2).argmax(2)

            inputLengths = torch.full(size = (images.size(0),), 
                                      fill_value = 128, 
                                      dtype=torch.long)

            # Compute Loss
            loss_rnn = loss_fn(pred_rnn, targets_output, inputLengths, targets_length)
            loss_ctc = loss_fn(pred_ctc, targets_output, inputLengths, targets_length)
            
            loss = loss_rnn + 0.1 * loss_ctc

            # Backward Pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train += loss.item()

            # Compute accuracy
            acc_train += (pred_int == targets_output).sum().item() / (targets_output.size(0) * targets_output.size(1))


        
        # Testing Loop
        loss_test = 0.0
        acc_test = 0.0
        with torch.no_grad():
            for images, targets in testing_loader:
                targets_output, targets_length = targets
                model.eval()

                images = images.to(DEVICE)
                
                # Forward Pass
                pred_rnn, pred_ctc = model(images)
                #shape = (BatchSize, SeqLen, NumClasses)
                # Convert to (SeqLen, BatchSize, NumClasses) for loss
                pred_rnn = pred_rnn.permute(1, 0, 2).log_softmax(2)
                pred_ctc = pred_ctc.permute(1, 0, 2).log_softmax(2)

                inputLengths = torch.full(size = (images.size(0),),
                                            fill_value = 128,
                                            dtype=torch.long)
                
                # Compute Loss
                loss_rnn = loss_fn(pred_rnn, targets_output, inputLengths, targets_length)
                loss_ctc = loss_fn(pred_ctc, targets_output, inputLengths, targets_length)

                loss = loss_rnn + 0.1 * loss_ctc
                loss_test += loss.item()

                # Compute accuracy
                pred_int = pred_rnn.permute(1, 0, 2).argmax(2)
                acc_test += (pred_int == targets_output).sum().item() / (targets_output.size(0) * targets_output.size(1))



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
        "rnn_dropout": [0.1, 0.25, 0.5],
        "conv_dropout": [0.1, 0.25, 0.5],
    }
    paramList = list(itertools.product(*parameters.values()))

    best_loss = 1000000
    best_params = [0.0001, (0.9, 0.999), 0.25, 0.25]
    """
    # Train different models for multiple folds
    for params in paramList:
        print("Training with parameters: ", params)
        learning_rate, beta, rnn_dropout, conv_dropout = params

        losses = []
        for train_idx, test_idx in kFolds.split(train_data):
            # Set up CRNN model + optimizer
            model = Recurrent_CNN(Data.n_chars, 
                                  rnn_dropout = rnn_dropout,
                                  Conv_dropout = conv_dropout
                                  )

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=beta)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

            # Set up data for the current fold
            train_data_fold = Subset(train_data, train_idx)
            test_data_fold = Subset(train_data, test_idx)
            train_loader = DataLoader(train_data_fold, batch_size = BATCH_SIZE, shuffle = True)
            test_loader = DataLoader(test_data_fold, batch_size = BATCH_SIZE, shuffle = False)

            #
            history = train_model(model = model,
                                  training_loader = train_loader,
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
    """
    ##########################################################################################################
    # Train the best model with all the training data and test on testing data
    # Save the performance and model at the end
    ##########################################################################################################
    print("Best model found with parameters: ", best_params)
    learning_rate, beta, rnn_dropout, conv_dropout = best_params
    model = Recurrent_CNN(Data.n_chars,
                          rnn_dropout = rnn_dropout,
                          Conv_dropout = conv_dropout
                          )
    #model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=beta)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    history = train_model(model = model,
                          training_loader = train_loader,
                          testing_loader = test_loader,
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
