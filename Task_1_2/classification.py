import numpy as np
from utils import *
import networks
import augmentation
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
def load_test_data(model_name):
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]
    
    classifier = networks.ClassifierCNN().to(device)
    classifier.load_state_dict(torch.load(model_name+".pth"))

    images, labels, _ = load_segmented_data(test_files=np.load("test_files.npy"), test=True)
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)

    return classes, classifier, images, labels

def load_training_data():
    images, labels, _ = load_segmented_data(test_files=np.load("test_files.npy"))
    encoder = load("encoder.joblib")
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)
    return images, labels

def plot_final(losses):
    plt.ioff()
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()

# https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def save_model(name, model, opt, n_patches, radius):
    name += "_n"+str(n_patches)+"_r"+str(radius)
    torch.save(model.state_dict(), uniquify(name+".pth"))
    torch.save(opt.state_dict(), uniquify(name+"_opt.pth"))

def train_batch(images, labels, recognizer, recognizer_opt, aug_S2_batch, S2_probs, agent_opt, augment=True):
    recognizer_opt.zero_grad()
    outputs = recognizer(images)
    recognizer_loss = RECOGNIZER_LOSS_FUNCTION(outputs, labels)
    recognizer_loss.backward()
    torch.nn.utils.clip_grad_value_(recognizer.parameters(), 100)
    recognizer_opt.step()

    if augment:
        outputs_S2 = recognizer(aug_S2_batch)
        augmentation.train(outputs, outputs_S2, labels, agent_opt, S2_probs)

    return recognizer_loss.item()

def train_epoch(images, labels, recognizer, recognizer_opt, aug_S2, S2_probs, agent_opt, augment=True):
    trainset = torch.utils.data.TensorDataset(images, aug_S2, S2_probs, labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    running_loss = 0.0
    for images, aug_S2_batch, S2_probs_batch, labels in trainloader:
        images, labels, aug_S2_batch = images.to(device), labels.to(device), aug_S2_batch.to(device)
        loss = train_batch(images, labels, recognizer, recognizer_opt, aug_S2_batch, S2_probs_batch, agent_opt, augment=augment)
        running_loss += loss

    return float(running_loss)

def train(n_patches, radius, augment=True):

    org_images, labels = load_training_data()

    agent = networks.AugmentAgentCNN(n_patches).to(device)
    agent_opt = optim.Adadelta(agent.parameters())
    recognizer = networks.ClassifierCNN().to(device)
    recognizer_opt = optim.AdamW(recognizer.parameters(), amsgrad=True)
    # summary(agent, input_size=(1, 1, h, w), device=device)
    # summary(recognizer, input_size=(1, 1, h, w), device=device)

    losses = []
    plt.ion()
    line, = plt.plot([], [])

    for ep in range(N_EPOCHS):
        if augment:
            images, aug_S2, S2_probs = augmentation.augment_data(org_images, agent, n_patches, radius)
            loss = train_epoch(images, labels, recognizer, recognizer_opt, aug_S2, S2_probs, agent_opt, augment=augment)
        else:
            loss = train_epoch(org_images, labels, recognizer, recognizer_opt, org_images, torch.rand(len(images),2*(n_patches+1),2), agent_opt, augment=True)

        losses.append(float(loss))
        plot_loss(float(loss), line, ep)

    save_model('recognizer', recognizer, recognizer_opt, n_patches, radius)
    save_model('agent', agent, agent_opt, n_patches, radius)
    plot_final(losses)

def test_classifier(model_name):
    classes, classifier, images, labels = load_test_data(model_name)

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    testnset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(testnset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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

def grid_search():
    # concept
    for n_patches in [1,2,3]:
        for radius in [5,10,20]:
            train(n_patches, radius)

def main():
    # load_segmented_data(print_mean_dims=True)

    # classes = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel',
    #            'He', 'Het', 'Kaf', 'Kaf-finala', 'Lamed',
    #            'Mem', 'Mem-medial', 'Nun-final', 'Nun-medial', 'Pe',
    #            'Pe-final', 'Qof', 'Resh', 'Samekh', 'Shin',
    #            'Taw', 'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw',
    #            'Yod', 'Zayin']

    if not os.path.exists("test_files.npy") or not os.path.exists("encoder.joblib"): prepare_train_test_data()
    # grid_search()
    # if not os.path.exists("recognizer.pth"): train(n_patches=1, radius=10, augment=True)
    # test_classifier('recognizer_n1_r10')
    test_classifier('recognizer_n1_r10_augmented')


##############################################
# Main script to look for the best hyperparameters
##############################################
if __name__ == '__main__':
    # print("Searching for the best hyperparameters for the CNN model")
    main()



# NO AUGMENTATION

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



# AUGMENTATION

# Accuracy: 92 %
# Accuracy for class Alef is 100.0 %
# Accuracy for class Ayin is 100.0 %
# Accuracy for class Bet is 90.0 %
# Accuracy for class Dalet is 55.6 %
# Accuracy for class Gimel is 93.3 %
# Accuracy for class He is 90.0 %
# Accuracy for class Het is 100.0 %
# Accuracy for class Kaf is 94.7 %
# Accuracy for class Kaf-final is 100.0 %
# Accuracy for class Lamed is 96.6 %
# Accuracy for class Mem is 93.3 %
# Accuracy for class Mem-medial is 96.7 %
# Accuracy for class Nun-final is 84.6 %
# Accuracy for class Nun-medial is 93.3 %
# Accuracy for class Pe is 25.0 %
# Accuracy for class Pe-final is 100.0 %
# Accuracy for class Qof is 96.3 %
# Accuracy for class Resh is 75.0 %
# Accuracy for class Samekh is 80.0 %
# Accuracy for class Shin is 96.7 %
# Accuracy for class Taw is 100.0 %
# Accuracy for class Tet is 96.7 %
# Accuracy for class Tsadi-final is 100.0 %
# Accuracy for class Tsadi-medial is 93.3 %
# Accuracy for class Waw is 75.0 %
# Accuracy for class Yod is 0.0 %
# Accuracy for class Zayin is 0.0 %