import numpy as np
from utils import *
import networks
import augmentation
import torch
from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from joblib import load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################
# CLASSES AND FUNCTIONS FOR THE CLASSIFICATION SECTION
##############################################
def load_test_data(model_name=''):
    """
    Get the test images and labels.
    @param model_name: If model_name (filename of the model without .pth) is given, the classifier will be returned as well.
    @return list of possible classes, labels, testloader for looping, and optionally the classifier
    """
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]
    
    if model_name:
        classifier = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
        classifier.load_state_dict(torch.load(model_name+".pth"))

    images, labels, _ = load_segmented_data(test_files=np.load("test_files.npy"), test=True)
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)

    testset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    if model_name: return classes, labels, testloader, classifier
    return classes, labels, testloader

def load_training_data():
    """
    Get the train images and labels.
    """
    images, labels, _ = load_segmented_data(test_files=np.load("test_files.npy"))
    encoder = load("encoder.joblib")
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)
    return images, labels

def plot_final(losses, ylabel):
    """
    Simple plot, given variable ylabel.
    """
    plt.figure()
    plt.xlabel('Batch')
    plt.ylabel(ylabel)
    plt.plot(losses)

def uniquify(path, find=False):
    """
    Adapt the filename so that it doesn't overwrite.
    @param path: Path to possible modify.
    @param find: Instead of modifying path, find the name of the latest version.
    @return (Possibly) modified pathname such as 'file.ext' 'file (1).ext' or 'file (2).ext'
            If find is True, then it returns the name of the latest version.
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    if find:
        if counter == 2: return filename + extension
        elif counter > 2: return filename + " (" + str(counter-2) + ")" + extension
    return path

def get_model_save_name(model_type, n_patches=None, radius=None):
    """
    Get the name of a model given certain parameters.
    @param model_type: 'recognizer' or 'agent'
    @param n_patches: Number of patches used for augmentation. Don't suppy to return
                    the recognizer trained without augmentation.
    @param radius: Radius used for augmentation
    @return Model filename (without extension or version such as '(1)')
    """
    if n_patches is None: return f"{model_type}"
    return f"{model_type}_n{n_patches}_r{radius}"

def save_model(name, model, opt, n_patches=None, radius=None):
    """
    Save model weights and optimizer to file without overwriting.
    @param name: 'recognizer' or 'agent'
    @param model: torch.nn model
    @param opt: optimizer
    @param n_patches: Number of patches used for augmentation. Don't suppy to save
                    recognizer trained without augmentation.
    @param radius: Radius used for augmentation
    """
    save_name = get_model_save_name(name, n_patches=n_patches, radius=radius)
    torch.save(model.state_dict(), uniquify(save_name+".pth"))
    torch.save(opt.state_dict(), uniquify(save_name+"_opt.pth"))

def train_batch(images, labels, recognizer, recognizer_opt, aug_S2_batch, agent_outputs, S, S2, agent_opt, augment=True, train_recog=True):
    """
    Train with a batch of images.
    @return Recognizer and augmentation agent loss (float)
    """
    outputs = recognizer(images)

    if train_recog:
        recognizer_opt.zero_grad()
        recognizer_loss = RECOGNIZER_LOSS_FUNCTION(outputs, labels)
        recognizer_loss.backward()
        torch.nn.utils.clip_grad_value_(recognizer.parameters(), 100)
        recognizer_opt.step()
        rec_loss = recognizer_loss.item()
    else:
        rec_loss = 0.0

    if augment:
        outputs_S2 = recognizer(aug_S2_batch)
        aug_loss = augmentation.train(outputs, outputs_S2, labels, agent_opt, agent_outputs, S, S2)
    else:
        aug_loss = 0.0

    return rec_loss, aug_loss

def train_epoch(images, labels, recognizer, recognizer_opt, agent_opt, aug_loss_line, rec_loss_line, acc_line, classes, test_labels, testloader, agent, n_patches, radius, augment=True, train_recog=True):
    """
    Train an episode.
    @param images: All (non-augmented) train images
    @param labels: True classes for the train set
    @param recognizer: Recognizer/classifier model
    @param recognizer_opt: Recognizer optimizer
    @param agent_opt: (Learning augmentation) agent optimizer
    @param aug_loss_line: Augmentation loss line
    @param rec_loss_line: Recognizer loss line
    @param acc_line: Accuracy line
    @param classes: Classes (order given by the encoder)
    @param test_labels: True classes for the test set
    @param testloader: Used to loop over test data
    @param agent: Augmentation agent model
    @param n_patches: Number of patches used for augmentation
    @param radius: Radius of patches used for augmentation
    @param augment: Whether the augmentation agent should be trained and applied
    @param train_recog: Whether the recognizer should be trained
    
    @return augmentation and recognizer losses, and accuracy for every batch
    """
    trainset = torch.utils.data.TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    batch_losses, batch_aug_losses, batch_accs = [], [], []

    for images, labels in trainloader:
        if augment:
            images, aug_S2, agent_outputs, S, S2 = augmentation.augment_data(images, agent, n_patches, radius)
            aug_S2 = aug_S2.to(device)
        else: aug_S2, agent_outputs, S, S2 = None, None, None, None

        images, labels = images.to(device), labels.to(device)
        batch_loss, batch_aug_loss = train_batch(images, labels, recognizer, recognizer_opt, aug_S2, agent_outputs, S, S2, agent_opt, augment=augment, train_recog=train_recog)

        correct, total, _, _ = test_classifier(classes, recognizer, test_labels, testloader)
        batch_acc = correct / total

        plot_interactive(batch_aug_loss, batch_loss, batch_acc, aug_loss_line, rec_loss_line, acc_line)
        batch_losses.append(batch_loss); batch_aug_losses.append(batch_aug_loss); batch_accs.append(batch_acc)

    return batch_losses, batch_aug_losses, batch_accs

def init_interactive():
    """
    Initialize interactive plots
    @return augmentation loss line, recognizer loss line, accuracy line
    """
    plt.ion()
    plt.figure("aug", figsize=(5, 3))
    aug_loss_line, = plt.plot([], [])
    plt.figure("rec", figsize=(5, 3))
    rec_loss_line, = plt.plot([], [])
    plt.figure("acc", figsize=(5, 3))
    acc_line, = plt.plot([], [])
    return aug_loss_line, rec_loss_line, acc_line

def train(n_patches=None, radius=None, augment=True, train_recog=True):
    """
    Train augmentation agent and/or recognizer.
    @param n_patches: Number of patches used for augmentation.
    @param radius: Radius used for augmentation
    @param augment: Whether there should be augmentation
    @param train_recog: Whether the recognizer should be trained (otherwise it is loaded)
    """
    org_images, labels = load_training_data()

    # train on subset of data
    # idxs = np.random.randint(org_images.shape[0], size=100)
    # idxs = [0]
    # org_images = org_images[idxs]
    # labels = labels[idxs]

    if augment:
        if n_patches == 4: n_points = 5
        else: n_points = 2*(n_patches+1)

        agent = networks.AugmentAgentCNN(n_points).to(device)
        agent_opt = optim.Adadelta(agent.parameters())
        # summary(agent, input_size=(1, 1, h, w), device=device)

        augment_path = os.path.join("img", "augmented")
        os.makedirs(augment_path, exist_ok=True)
    else: agent, agent_opt = None, None


    if train_recog:
        recognizer = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
        recognizer_opt = optim.AdamW(recognizer.parameters(), amsgrad=True)
        # summary(recognizer, input_size=(1, 1, CHARACTER_HEIGHT, CHARACTER_WIDTH), device=device)
        classes, test_labels, testloader = load_test_data()
    else:
        recognizer_opt = None
        classes, test_labels, testloader, recognizer = load_test_data(uniquify(get_model_save_name('recognizer', n_patches=n_patches, radius=radius), find=True))
    recognizer = recognizer.to(device)

    aug_loss_line, rec_loss_line, acc_line = init_interactive()
    losses, aug_losses, accs = [], [], []

    for ep in range(N_EPOCHS):
        if augment:
            batch_losses, batch_aug_losses, batch_accs = train_epoch(org_images, labels, recognizer, recognizer_opt, agent_opt, aug_loss_line, rec_loss_line, acc_line, classes, test_labels, testloader, agent, n_patches, radius, augment=True, train_recog=train_recog)

            # save example augmented images
            ex_images = org_images[[0, 500, 1000]].to(device)
            agent_outputs = agent(ex_images)
            ex_images = torch.squeeze(ex_images, 1).detach().cpu().numpy()
            S = torch.max(agent_outputs, 3).indices.detach().cpu().numpy()
            for i in range(ex_images.shape[0]):
                example_dist, src_pts, dst_pts = augmentation.distort(ex_images[i], n_patches, radius, S[i], return_points=True, max_radius=True)
                example = augmentation.draw_augment_arrows(ex_images[i], src_pts, dst_pts, radius)
                cv2.imwrite(os.path.join(augment_path, f"{i}_e{ep}.png"), example)
                cv2.imwrite(os.path.join(augment_path, f"{i}_e{ep}_dist.png"), example_dist)
        else:
            batch_losses, batch_aug_losses, batch_accs = train_epoch(org_images, labels, recognizer, recognizer_opt, agent_opt, aug_loss_line, rec_loss_line, acc_line, classes, test_labels, testloader, agent, n_patches, radius, augment=False, train_recog=True)

        losses.extend(batch_losses); aug_losses.extend(batch_aug_losses); accs.extend(batch_accs)

    if train_recog: save_model('recognizer', recognizer, recognizer_opt, n_patches, radius)
    if augment: save_model('agent', agent, agent_opt, n_patches, radius)
    plt.ioff()
    plot_final(losses, 'Loss'); plot_final(accs, 'Accuracy'); plot_final(aug_losses, 'Loss')
    plt.show()

def load_and_test(model_name):
    """
    Load and test a classier on the test set.
    @param model_name: Filename of the model without .pth
    """
    classes, labels, testloader, classifier = load_test_data(model_name)
    
    correct, total, correct_pred, total_pred = test_classifier(classes, classifier, labels, testloader)

    print(f'Accuracy: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname.decode()} is {accuracy:.1f} % ({correct_count}/{total_pred[classname]})')

def test_classifier(classes, classifier, labels, testloader):
    """
    Test a classier on the test set.
    @param classes: Classes (order given by the encoder)
    @param classifier: Classifier model
    @param labels: True classes of the test images
    @param testloader: Used to loop over test data
    
    @return Number of overall correct, total tested, dict of correct predictions per class, dict of total predictions per class
    """
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

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

    return correct, total, correct_pred, total_pred

def main():
    # show_image_dimensions()
    
    if not os.path.exists("test_files.npy") or not os.path.exists("encoder.joblib"): prepare_train_test_data()
    
    # n_patches, radius = 1, 10
    n_patches, radius = None, None
    recognizer_name = uniquify(get_model_save_name('recognizer', n_patches=n_patches, radius=radius), find=True)
    # if not os.path.exists(recognizer_name + '.pth'):
    train(n_patches=n_patches, radius=radius, augment=False, train_recog=True)

    load_and_test(recognizer_name)


##############################################
# Main script to look for the best hyperparameters
##############################################
if __name__ == '__main__':
    # print("Searching for the best hyperparameters for the CNN model")
    main()