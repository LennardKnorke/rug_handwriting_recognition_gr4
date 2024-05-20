import numpy as np
import utils
import networks
import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from joblib import load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################
# CLASSES AND FUNCTIONS FOR THE CLASSIFICATION SECTION
##############################################
def train():
    batch_size = 64
    epochs = 50

    x, y, _ = utils.load_segmented_data(test_files=np.load("test_files.npy"))
    encoder = load("encoder.joblib")
    y = encoder.transform(y[:, np.newaxis])
    x = torch.Tensor(x); y = torch.Tensor(y)

    # agent = networks.AugmentAgentCNN(n_patches).to(device)
    # summary(agent, input_size=(1, 1, h, w), device=device)
    # agent_opt = optim.Adadelta(agent.parameters())
    recognizer = networks.ClassifierCNN().to(device)
    # summary(recognizer, input_size=(1, 1, h, w), device=device)
    recognizer_opt = optim.AdamW(recognizer.parameters(), amsgrad=True)
    recognizer_loss_func = nn.CrossEntropyLoss()

    losses = []
    plt.ion()
    line, = plt.plot([], [])

    for ep in range(epochs):
        trainset = torch.utils.data.TensorDataset(x,y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            recognizer_opt.zero_grad()
            outputs = recognizer(inputs)
            loss = recognizer_loss_func(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(recognizer.parameters(), 100)
            recognizer_opt.step()

            running_loss += loss.item()

        losses.append(float(running_loss))
        utils.plot_loss(float(running_loss), line, ep)

    torch.save(recognizer.state_dict(), "recognizer.pth")
    torch.save(recognizer_opt.state_dict(), "recognizer_opt.pth")
    plt.ioff()
    
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()

def test_classifier():
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]
    
    classifier = networks.ClassifierCNN().to(device)
    classifier.load_state_dict(torch.load("recognizer.pth"))

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    x, y, _ = utils.load_segmented_data(test_files=np.load("test_files.npy"), test=True)
    y = encoder.transform(y[:, np.newaxis])
    x = torch.Tensor(x); y = torch.Tensor(y)

    batch_size = 64
    testnset = torch.utils.data.TensorDataset(x,y)
    testloader = torch.utils.data.DataLoader(testnset, batch_size=batch_size, shuffle=True, num_workers=2)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            predicted = torch.max(outputs, 1).indices
            labels = torch.max(labels, 1).indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname.decode()} is {accuracy:.1f} %')

def main():
    # utils.load_segmented_data(print_mean_dims=True)

    # classes = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel',
    #            'He', 'Het', 'Kaf', 'Kaf-finala', 'Lamed',
    #            'Mem', 'Mem-medial', 'Nun-final', 'Nun-medial', 'Pe',
    #            'Pe-final', 'Qof', 'Resh', 'Samekh', 'Shin',
    #            'Taw', 'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw',
    #            'Yod', 'Zayin']

    if not os.path.exists("test_files.npy") or not os.path.exists("encoder.joblib"): utils.prepare_train_test_data()
    if not os.path.exists("recognizer.pth"): train()
    test_classifier()


##############################################
# Main script to look for the best hyperparameters
##############################################
if __name__ == '__main__':
    # print("Searching for the best hyperparameters for the CNN model")
    main()




# Accuracy: 92 %
# Accuracy for class Alef is 96.7 %
# Accuracy for class Ayin is 100.0 %
# Accuracy for class Bet is 93.3 %
# Accuracy for class Dalet is 55.6 %
# Accuracy for class Gimel is 90.0 %
# Accuracy for class He is 100.0 %
# Accuracy for class Het is 96.7 %
# Accuracy for class Kaf is 89.5 %
# Accuracy for class Kaf-final is 0.0 %
# Accuracy for class Lamed is 93.1 %
# Accuracy for class Mem is 93.3 %
# Accuracy for class Mem-medial is 100.0 %
# Accuracy for class Nun-final is 84.6 %
# Accuracy for class Nun-medial is 96.7 %
# Accuracy for class Pe is 75.0 %
# Accuracy for class Pe-final is 0.0 %
# Accuracy for class Qof is 96.3 %
# Accuracy for class Resh is 50.0 %
# Accuracy for class Samekh is 93.3 %
# Accuracy for class Shin is 100.0 %
# Accuracy for class Taw is 100.0 %
# Accuracy for class Tet is 93.3 %
# Accuracy for class Tsadi-final is 100.0 %
# Accuracy for class Tsadi-medial is 96.7 %
# Accuracy for class Waw is 83.3 %
# Accuracy for class Yod is 0.0 %
# Accuracy for class Zayin is 0.0 %
