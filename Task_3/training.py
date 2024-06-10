# Import third party modules
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

# Our modules
import augmentation as augment
from dataset import *
from network import *
from utils import *

# Makros only relevant for training
BATCH_SIZE = 8
N_EPOCHS = 10
N_FOLDS = 5
TEST_RATIO = 0.2

def train_model(model : nn.Module, 
                optimizer : optim,
                scheduler : optim.lr_scheduler,

                training_loader : DataLoader,
                testing_loader : DataLoader,

                print_epochs : bool = False,
                augmentation : dict = {"type": None}
                )-> Tuple[list, list, list, list, list, list]:
    """
    Train a model on a dataset and run validation with given optimization/scheduler
    @param model: torch model to train
    @param training_loader: loader for the training data
    @param testing_loader: loader for the validation data
    @param optimizer: optimizer to use for training
    @param scheduler: scheduler to use for training
    @return: 6 lists, 3 for training 3 for testing, each containing the loss, word error rate, and character error rate
    """
    assert augmentation["type"] in [None, "simple", "RL", "both"], "Invalid augmentation type"
    train_loss = []
    train_wer = []
    train_cer = []
    train_aug_loss = []

    test_loss = []
    test_wer = []
    test_cer = []

    model.to(DEVICE)

    if augmentation["type"] == "RL" or augmentation["type"] == "both":
        augmentation["model"].to(DEVICE)
    
    for ep in tqdm(range(N_EPOCHS), desc='Epochs'):
        loss_train_ep = 0.0
        wer_train_ep = 0.0
        cer_train_ep = 0.0
        if augmentation["type"] == "RL" or augmentation["type"] == "both":
            aug_loss_ep = 0.0
        loss_fn = nn.CTCLoss(zero_infinity = True)

        # Training Loop
        model.train()
        if augmentation["type"] == "RL" or augmentation["type"] == "both":
            augmentation["model"].train()

        batch_progress = tqdm(train_loader, desc='Batches', leave=False)
        for i, (images, targets) in enumerate(batch_progress):
            optimizer.zero_grad()
            # Set up targets
            targets_encoded, targets_length, targets_strings = targets      
            
            # Load (raw) greyscaled images
            images = load_image_batch(images)
            
            # Augmentation step
            if augmentation["type"] == "simple" or augmentation["type"] == "both":
                pass
                # images = simple_augmentation(images)
            if augmentation["type"] == "RL" or augmentation["type"] == "both":
                images, aug_S2, agent_outputs, S, S2 = augment.augment_data(images, augmentation["n_patches"], augmentation["radius"], agent=augmentation["model"])
                aug_S2 = preprocess_batch(aug_S2)
                aug_S2 = aug_S2.to(DEVICE)

            # Resize and normalize images for the model
            images = preprocess_batch(images)
            images = images.to(DEVICE)

            # Forward Pass
            pred_rnn, pred_ctc = model(images)                          # shape = (BatchSize, SeqLen, NumClasses)
            pred_int = pred_rnn.softmax(2).argmax(2)                    # Shape = (BatchSize, SeqLen), with integers for each pred char (0 = blank)

            # Convert to (SeqLen, BatchSize, NumClasses) for loss
            pred_rnn = pred_rnn.permute(1, 0, 2).log_softmax(2)         
            pred_ctc = pred_ctc.permute(1, 0, 2).log_softmax(2)

            inputLengths = torch.full(size = (images.size(0),), 
                                      fill_value = MAX_SEQ_LENGTH, 
                                      dtype=torch.int32)

            # Compute Loss
            loss = loss_fn(pred_rnn.cpu(), targets_encoded, inputLengths, targets_length)            
            loss += 0.1 * loss_fn(pred_ctc.cpu(), targets_encoded, inputLengths, targets_length)

            # Backward Pass
            loss.backward()
            optimizer.step()
            loss_train_ep += loss.item()

            # Compute accuracies
            decoded_strings = ctc_decode(pred_int)

            wer, cer = get_error_rates(decoded_strings, targets_strings)
            wer_train_ep += wer
            cer_train_ep += cer

            # Augmentation Training step
            if augmentation["type"] == "RL" or augmentation["type"] == "both":
                pred_rnn_S2, _ = model(aug_S2)
                pred_int_S2 = pred_rnn_S2.softmax(2).argmax(2)
                pred_rnn_S2 = pred_rnn_S2.permute(1, 0, 2).log_softmax(2)
                decoded_strings_S2 = ctc_decode(pred_int_S2)

                # print(decoded_strings[i], "-----", decoded_strings_S2[i])

                error = torch.FloatTensor(np.array([fastwer.score_sent(decoded_strings[i], targets_strings[i], char_level=True) for i in range(len(targets_strings))]))
                error_S2 = torch.FloatTensor(np.array([fastwer.score_sent(decoded_strings_S2[i], targets_strings[i], char_level=True) for i in range(len(targets_strings))]))

                aug_loss = augment.train(error = error,
                    error_S2 = error_S2,
                    agent_opt = augmentation["opt"],
                    agent_outputs = agent_outputs,
                    S = S,
                    S2 = S2)
                
                # plt.figure(figsize=(15, 5))
                # plt.subplot(1, 2, 1)
                # plt.imshow(images[0][0].cpu())
                # plt.axis('off')
                # plt.subplot(1, 2, 2)
                # plt.imshow(aug_S2[0][0].cpu())
                # plt.axis('off')
                # plt.show()
            
                aug_loss_ep += aug_loss

            batch_progress.set_postfix(loss=loss_train_ep/(i+1), wer=wer_train_ep/(i+1), cer=cer_train_ep/(i+1))

        train_loss.append(loss_train_ep / len(training_loader))
        train_wer.append(wer_train_ep / len(training_loader))
        train_cer.append(cer_train_ep / len(training_loader))
        if augmentation["type"] == "RL" or augmentation["type"] == "both":
            train_aug_loss.append(aug_loss_ep / len(training_loader))

        
        # Testing Loop
        loss_test_ep = 0.0
        wer_test_ep = 0.0
        cer_test_ep = 0.0
        model.eval()
        with torch.no_grad():
            for images, targets in testing_loader:
                # Set up targets
                targets_encoded, targets_length, targets_strings = targets

                # Preprocess batch (no augmentation in evaluation)
                images = preprocess_batch(load_image_batch(images))
                images = images.to(DEVICE)
                
                # Forward Pass
                pred_rnn, pred_ctc = model(images)
                pred_int = pred_rnn.softmax(2).argmax(2)            # Shape = (BatchSize, SeqLen), with integers for each pred char (0 = blank)

                # Convert to (SeqLen, BatchSize, NumClasses) for loss
                pred_rnn = pred_rnn.permute(1, 0, 2).log_softmax(2)
                pred_ctc = pred_ctc.permute(1, 0, 2).log_softmax(2)

                inputLengths = torch.full(size = (images.size(0),),
                                            fill_value = MAX_SEQ_LENGTH,
                                            dtype=torch.long)
                
                # Compute Loss
                loss = loss_fn(pred_rnn.cpu(), targets_encoded, inputLengths, targets_length)            
                #loss += 0.1 * loss_fn(pred_ctc.cpu(), targets_encoded, inputLengths, targets_length)
                # AUTHORS SUGGESTED OMMITING THE CTC OUTPUT BEYOND TRAINING
                loss_test_ep += loss.item()

                # Compute testing error rates
                decoded_strings = ctc_decode(pred_int)
                wer, cer = get_error_rates(decoded_strings, targets_strings)
                wer_test_ep += wer
                cer_test_ep += cer
        # End of Epoch
        scheduler.step()
        test_loss.append(loss_test_ep / len(testing_loader))
        test_wer.append(wer_test_ep / len(testing_loader))
        test_cer.append(cer_test_ep / len(testing_loader))

        if print_epochs:
            print(f"\nEp {ep}. Train Loss: {train_loss[-1]}. Train WER: {train_wer[-1]}. Train CER: {train_cer[-1]}. Test Loss: {test_loss[-1]}. Test WER: {test_wer[-1]}. Test CER: {test_cer[-1]}")

    if augmentation["type"] == "RL" or augmentation["type"] == "both":
        return train_loss, test_loss, train_wer, test_wer, train_cer, test_cer, train_aug_loss
    return train_loss, test_loss, train_wer, test_wer, train_cer, test_cer

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
        "conv_dropout": [0.0, 0.1, 0.25],        # Might not be needed in case we figure our THEIR code
        "augment_dicts": [
            {"type": "simple"},
            {"tye" : None}
        ]
    }
    paramList = list(itertools.product(*parameters.values()))

    best_loss = 1000000
    best_params = [0.0, 2, 10]
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
    conv_dropout, n_patches, radius = best_params
    model = Recurrent_CNN(N_CHARS + 1,
                          Conv_dropout = conv_dropout
                          )
    
    # model.load_state_dict(torch.load("./Task_3/IAM_the_best_model.pth"))
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 180], gamma = 0.1)

    n_points = 2*(n_patches + 1)
    aug_model = AugmentAgentCNN(n_points)
    aug_optimizer = optim.Adadelta(aug_model.parameters())

    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    augmentation = {'model': aug_model, 'type': 'RL', 'opt': aug_optimizer, 'n_patches': n_patches, 'radius': radius}

    train_loss, test_loss, train_wer, test_wer, train_cer, test_cer, aug_loss = train_model(model = model,
                                                                                  training_loader = train_loader,
                                                                                  testing_loader = test_loader,
                                                                                  optimizer = optimizer,
                                                                                  scheduler = scheduler,
                                                                                  print_epochs = True,
                                                                                  augmentation = augmentation)
    print(f"Final Results: \nTest Loss:{test_loss[-1]} \nTest Word Error Rate: {test_wer[-1]}\nTest Character Error Rate: {test_cer[-1]}")
    # Save the model
    torch.save(model.state_dict(), "./Task_3/IAM_the_best_model.pth")
    torch.save(optimizer.state_dict(), "./Task_3/IAM_the_best_model_optim.pth")
    if augmentation["type"] == "RL" or augmentation["type"] == "both":
        torch.save(aug_model.state_dict(), "./Task_3/IAM_the_best_aug.pth")
        torch.save(aug_optimizer.state_dict(), "./Task_3/IAM_the_best_aug_optim.pth")
    
    # Plot performance
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(test_loss, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("./Task_3/loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(test_cer, label="Testing CER")
    plt.plot(train_cer, label="Training CER")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Character Error Rate")
    plt.savefig("./Task_3/cer_plot.png")
    plt.close()

    plt.figure()
    plt.plot(test_wer, label="Testing WER")
    plt.plot(train_wer, label="Training WER")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Word Error Rate")
    plt.savefig("./Task_3/wer_plot.png")
    plt.close()

    if augmentation["type"] == "RL" or augmentation["type"] == "both":
        plt.figure()
        plt.plot(aug_loss)
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Augmentation agent loss")
        plt.savefig("./Task_3/aug_loss_plot.png")
        plt.close()
    # Done