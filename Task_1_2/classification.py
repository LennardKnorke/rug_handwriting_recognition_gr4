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
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################
# CLASSES AND FUNCTIONS FOR THE CLASSIFICATION SECTION
##############################################
def load_classifier(model_name):
    classifier = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
    classifier.load_state_dict(torch.load(os.path.join("models", f"{model_name}.pth")))
    return classifier


def load_test_data(batch_size, split, model_name=''):
    """
    Get the test images and labels.
    @param model_name: If model_name (filename of the model without .pth) is given, the classifier will be returned as well.
    @return list of possible classes, labels, testloader for looping, and optionally the classifier
    """
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]
    
    if model_name: classifier = load_classifier(model_name)

    test_files = None
    if split: test_files = np.load(get_test_split_filename(split))

    images, labels, _ = load_segmented_data(test_files=test_files, test=True)
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)

    testset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    if model_name: return classes, labels, testloader, classifier
    return classes, labels, testloader

def load_training_data(split=0):
    """
    Get the train images and labels.
    """
    test_files = None
    if split: test_files = np.load(get_test_split_filename(split))
    images, labels, _ = load_segmented_data(test_files=test_files)
    encoder = load("encoder.joblib")
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)
    return images, labels

def plot_final(losses, ylabel, ylim=None):
    """
    Simple plot, given variable ylabel.
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.gca().yaxis.tick_right()
    if ylim is not None: plt.gca().set_ylim(ylim)
    plt.plot(losses)

def get_model_save_name(model_type, n_patches=None, radius=None, N=None, M=None):
    """
    Get the name of a model given certain parameters.
    @param model_type: 'recognizer' or 'agent'
    @param n_patches: Number of patches used for elastic morphing.
    @param radius: Radius used for elastic morphing
    @param N: Number of transformations to apply on an image in a row for RandAugment
    @param M: Magnitude of transformations for RandAugment
    @return Model filename (without extension or version such as '(1)')
    """
    name = f"{model_type}"
    if n_patches is not None: name += f"_n{n_patches}_r{radius}"
    if N is not None: name += f"_N{N}_M{M}"
    return name

def save_model(name, model, opt, n_patches=None, radius=None, N=None, M=None):
    """
    Save model weights and optimizer to file without overwriting.
    @param name: 'recognizer' or 'agent'
    @param model: torch.nn model
    @param opt: optimizer
    @param n_patches: Number of patches used for augmentation.
    @param N: Number of transformations to apply on an image in a row for RandAugment
    @param M: Magnitude of transformations for RandAugment
    @param radius: Radius used for augmentation
    """
    save_name = get_model_save_name(name, n_patches=n_patches, radius=radius, N=N, M=M)
    os.makedirs("models", exist_ok=True)
    final_save_name = uniquify(os.path.join("models", f"{save_name}"))
    torch.save(model.state_dict(), f"{final_save_name}.pth")
    torch.save(opt.state_dict(), uniquify(os.path.join("models", f"{save_name}_opt.pth")))
    return final_save_name

def train_batch(images, labels, recognizer, recognizer_opt, aug_S2_batch, agent_outputs, S, S2, agent_opt, lta=True, train_recog=True):
    """
    Train with a batch of images.
    @return Recognizer and augmentation agent loss (float)
    """
    outputs = recognizer(images)
    recognizer_loss = RECOGNIZER_LOSS_FUNCTION(outputs, labels)

    if train_recog:
        recognizer_opt.zero_grad()
        recognizer_loss.backward()
        torch.nn.utils.clip_grad_value_(recognizer.parameters(), 100)
        recognizer_opt.step()
    rec_loss = recognizer_loss.item()

    if lta:
        outputs_S2 = recognizer(aug_S2_batch)
        aug_loss = augmentation.train(outputs, outputs_S2, labels, agent_opt, agent_outputs, S, S2)
    else:
        aug_loss = 0.0

    return rec_loss, aug_loss

def train_epoch(images, batch_size, labels, recognizer, recognizer_opt, agent_opt, classes, test_labels, testloader, agent, n_patches, radius, N, M, elastic=True, lta=True, train_recog=True, randaug_sampler=None, split=0):
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
    @param N: Number of transformations to apply on an image in a row for RandAugment
    @param M: Magnitude of transformations for RandAugment (0, 1, or 2)
    @param elastic: Whether the augmentation agent should be applied
    @param lta: Whether the agent should be trained. Otherwise, random augmentation is used.
    @param train_recog: Whether the recognizer should be trained
    @param randaug_sampler: USed to apply randaug to image (None if not using randaug)
    
    @return augmentation and recognizer losses, and accuracy for every batch
    """
    trainset = torch.utils.data.TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    loss, aug_loss, acc = 0.0, 0.0, 0.0

    batch_progress = tqdm(trainloader, desc='Batches', leave=False)
    for i, (images, labels) in enumerate(batch_progress):
        if elastic:
            if lta:
                images, aug_S2, agent_outputs, S, S2 = augmentation.augment_data(images, n_patches, radius, agent=agent)
                aug_S2 = aug_S2.to(device)
            else:
                images = augmentation.augment_data(images, n_patches, radius)
        if not lta: aug_S2, agent_outputs, S, S2 = None, None, None, None

        if randaug_sampler is not None:
            images = augmentation.randaug_data(images, randaug_sampler)

        images, labels = images.to(device), labels.to(device)
        batch_loss, batch_aug_loss = train_batch(images, labels, recognizer, recognizer_opt, aug_S2, agent_outputs, S, S2, agent_opt, lta=lta, train_recog=train_recog)

        if split:
            correct, total, _, _ = test_classifier(classes, recognizer, test_labels, testloader)
            batch_acc = correct / total
        else: batch_acc = 0.0

        loss += batch_loss; aug_loss += batch_aug_loss; acc += batch_acc
        batch_progress.set_postfix(loss=loss/(i+1), acc=acc/(i+1))

    return loss/len(trainloader), aug_loss/len(trainloader), acc/len(trainloader)

def train(n_epochs, batch_size, n_patches=None, radius=None, N=None, M=None, elastic=True, lta=False, train_recog=True, randaug=True, split=0):
    """
    Train augmentation agent and/or recognizer.
    @param n_patches: Number of patches used for augmentation.
    @param radius: Radius used for augmentation
    @param N: Number of transformations to apply on an image in a row for RandAugment
    @param M: Magnitude of transformations for RandAugment (0, 1, or 2)
    @param elastic: Whether there should be elastic augmentation
    @param lta: Whether the augmentation agent should be trained (otherwise random augmentation)
    @param train_recog: Whether the recognizer should be trained (otherwise it is loaded)
    @param randaug: Whether the randaug should be applied (using RandAugment policy)
    """
    org_images, labels = load_training_data(split)
    
    # train on subset of data
    # idxs = np.random.randint(org_images.shape[0], size=100)
    # idxs = [0]
    # org_images = org_images[idxs]
    # labels = labels[idxs]

    if elastic:
        if n_patches == 4: n_points = 5
        else: n_points = 2*(n_patches+1)

        agent = networks.AugmentAgentCNN(n_points).to(device)
        # summary(agent, input_size=(1, 1, h, w), device=device)
        if lta: agent_opt = optim.Adadelta(agent.parameters())
        else: agent_opt = None

        augment_path = os.path.join("img", "augmented")
        os.makedirs(augment_path, exist_ok=True)
    else: agent, agent_opt = None, None

    if randaug:
        randaug_sampler = augmentation.RandomAug(N, M)
    else: randaug_sampler = None

    if not split: classes, test_labels, testloader = None, None, None

    if train_recog:
        recognizer = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
        recognizer_opt = optim.AdamW(recognizer.parameters(), amsgrad=True)
        # summary(recognizer, input_size=(1, 1, CHARACTER_HEIGHT, CHARACTER_WIDTH), device=device)
        if split: classes, test_labels, testloader = load_test_data(batch_size, split)
    else:
        recognizer_opt = None
        if split: classes, test_labels, testloader, recognizer = load_test_data(batch_size, split, uniquify(get_model_save_name('recognizer', n_patches=n_patches, radius=radius, N=N, M=M), find=True))
    recognizer = recognizer.to(device)

    losses, aug_losses, accs = [], [], []

    for ep in tqdm(range(n_epochs), desc='Epochs'):
        if elastic:
            batch_losses, batch_aug_losses, batch_accs = train_epoch(org_images, batch_size, labels, recognizer, recognizer_opt, agent_opt, classes, test_labels, testloader, agent, n_patches, radius, N, M, elastic=True, lta=lta, train_recog=train_recog, randaug_sampler=randaug_sampler, split=split)

            if lta:
                # save example augmented images from agent
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
            ep_loss, ep_aug_loss, ep_acc = train_epoch(org_images, batch_size, labels, recognizer, recognizer_opt, agent_opt, classes, test_labels, testloader, agent, n_patches, radius, N, M, elastic=False, lta=False, train_recog=True, randaug_sampler=randaug_sampler, split=split)


        losses.append(ep_loss); aug_losses.append(ep_aug_loss); accs.append(ep_acc)
        print(f"\nEp {ep}. Train Loss: {ep_loss:.2f}. Test Accuracy: {ep_acc:.4f}.")

    if train_recog:
        save_name = save_model('recognizer', recognizer, recognizer_opt, n_patches, radius, N, M)
        plot_final(losses, 'Classifier Loss')
        plt.savefig(f"{save_name}_loss.png")
        plot_final(accs, 'Accuracy')
        plt.savefig(f"{save_name}_acc.png")
    if lta:
        save_name = save_model('agent', agent, agent_opt, n_patches, radius, N, M)
        plot_final(aug_losses, 'Agent Loss', ylim=[0,1])
        plt.savefig(f"{save_name}_aug_loss.png")
    plt.ioff()
    plt.show()

def load_and_test(batch_size, model_name, split=0):
    """
    Load and test a classier on the test set.
    @param model_name: Filename of the model without .pth
    """
    classes, labels, testloader, classifier = load_test_data(batch_size, split, model_name)
    
    correct, total, correct_pred, total_pred = test_classifier(classes, classifier, labels, testloader)

    print(f'Accuracy: {100 * correct / total:.2f} %')

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
    split = 10
    if not os.path.exists(get_test_split_filename(split)) or not os.path.exists("encoder.joblib"): create_test_split(split)
    
    n_patches, radius, N, M = 1, 10, 3, 1
    n_patches, radius, N, M = None, None, None, None
    n_epochs = 25
    batch_size = 64
    # n_patches, radius = None, None
    recognizer_name = uniquify(get_model_save_name('recognizer', n_patches, radius, N, M), find=True)
    if not os.path.exists(os.path.join("models", f"{recognizer_name}.pth")): train(n_epochs, batch_size, n_patches, radius, N, M, elastic=False, lta=False, train_recog=True, randaug=False, split=split)

    load_and_test(batch_size, recognizer_name, split)


##############################################
# Main script to look for the best hyperparameters
##############################################
if __name__ == '__main__':
    # print("Searching for the best hyperparameters for the CNN model")
    main()