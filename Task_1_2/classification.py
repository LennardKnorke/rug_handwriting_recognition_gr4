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


def load_classifier(model_name):
    """
    Load a classifier/recognizer model
    @param model_name: Path to the saved model file

    @return classifier model
    """
    classifier = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
    classifier.load_state_dict(torch.load(f"{model_name}.pth"))
    return classifier


def load_test_data(batch_size, split, model_name=''):
    """
    Get the test images and labels.
    @param batch_size: Batch size
    @param split: Test split percentage
    @param model_name: If model_name (filename of the model without .pth) is given, the classifier will be returned as well.

    @return list of possible classes, labels, testloader, and optionally the classifier
    """
    encoder = load("encoder.joblib")
    classes = encoder.categories_[0]
    
    if model_name: classifier = load_classifier(model_name)

    test_files = None
    if split:
        test_files = np.load(get_test_split_filename(split))

    images, labels, _ = load_segmented_data("img//segmented", test_files=test_files, test=True)
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)

    testset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    if model_name: return classes, labels, testloader, classifier
    return classes, labels, testloader

def load_training_data(split=0):
    """
    Get the train images and labels.
    @param split: Test split percentage

    @return Train images and character labels
    """
    test_files = None
    if split: test_files = np.load(get_test_split_filename(split))
    images, labels, _ = load_segmented_data("img//segmented", test_files=test_files)
    encoder = load("encoder.joblib")
    labels = encoder.transform(labels[:, np.newaxis])
    images = torch.Tensor(images); labels = torch.Tensor(labels)
    return images, labels

def plot_final(losses, ylabel, ylim=None):
    """
    Simple plot.
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.gca().yaxis.tick_right()
    if ylim is not None: plt.gca().set_ylim(ylim)
    plt.plot(losses)

def get_model_save_name(model_type, aug_dict):
    """
    Get the name of a model given certain parameters.
    @param model_type: 'recognizer' or 'agent'
    @param aug_dict: Contains:
                        n_patches: Number of patches used for elastic morphing.
                        radius: Radius used for elastic morphing.
                        N: Number of transformations to apply on an image in a row for RandomAugment
                        M: Magnitude of transformations for RandomAugment (0, 1, or 2)

    @return Model filename (without extension or version such as '(1)')
    """
    name = f"{model_type}"
    if aug_dict['n_patches'] is not None:
        name += f"_n{aug_dict['n_patches']}_r{aug_dict['radius']}"
    if aug_dict['N'] is not None:
        name += f"_N{aug_dict['N']}_M{aug_dict['M']}"
    return name

def save_model(name, model, opt, aug_dict):
    """
    Save model weights and optimizer to file without overwriting.
    @param name: 'recognizer' or 'agent'
    @param model: torch.nn model
    @param opt: optimizer
    @param aug_dict: Contains:
                        n_patches: Number of patches used for elastic morphing.
                        radius: Radius used for elastic morphing.
                        N: Number of transformations to apply on an image in a row for RandomAugment
                        M: Magnitude of transformations for RandomAugment (0, 1, or 2)

    @return Saved model filename
    """
    save_name = get_model_save_name(name, aug_dict)
    os.makedirs("models", exist_ok=True)
    models_save_name = uniquify(os.path.join("models", f"{save_name}"))
    torch.save(model.state_dict(), f"{models_save_name}.pth")
    torch.save(opt.state_dict(), uniquify(os.path.join("models", f"{save_name}_opt.pth")))

    torch.save(model.state_dict(), f"{save_name}.pth")
    return save_name

def train_batch(images, labels, lta_dict, recognizer, recognizer_opt, train_recog=True):
    """
    Train with a batch of images.
    @param images: augmented train images
    @param labels: True classes for the train set
    @param lta_dict: Contains:
                        lta: Whether the augmentation agent should be trained for elastic morphing (with Learn to Augment)
                        aug_S2: Images augmented by the random moving state (for Learn to Augment)
                        agent_opt: (Learning augmentation) agent optimizer
                        S: Moving state for the directions of movement for each fiducial point.
                        S2: Random moving state for the directions of movement for each fiducial point.
                        agent_outputs: Outputs of the augmentation agent network
    @param recognizer: Recognizer/classifier model
    @param recognizer_opt: Recognizer optimizer
    @param train_recog: Whether the recognizer should be trained

    @return batch classifier and augmentation loss
    """
    outputs = recognizer(images)
    recognizer_loss = RECOGNIZER_LOSS_FUNCTION(outputs, labels)

    if train_recog:
        recognizer_opt.zero_grad()
        recognizer_loss.backward()
        torch.nn.utils.clip_grad_value_(recognizer.parameters(), 100)
        recognizer_opt.step()
    rec_loss = recognizer_loss.item()

    aug_loss = 0.0
    # Train Learn to Augment agent
    if lta_dict['lta']:
        outputs_S2 = recognizer(lta_dict['aug_S2'])
        aug_loss = augmentation.train(outputs,
                                      outputs_S2,
                                      labels,
                                      lta_dict['agent_opt'],
                                      lta_dict['agent_outputs'],
                                      lta_dict['S'],
                                      lta_dict['S2'])
        

    return rec_loss, aug_loss

def train_epoch(images, batch_size, aug_dict, labels, recognizer, recognizer_opt, classes, test_labels, testloader, train_recog=True, split=0):
    """
    Train an episode.
    @param images: All (non-augmented) train images
    @param batch_size: Batch size
    @param aug_dict: Contains:
                        type: 'elastic' and/or 'randaug' for Elastic morphing and RandomAugment to be applied
                        n_patches: Number of patches used for elastic morphing.
                        radius: Radius used for elastic morphing.
                        N: Number of transformations to apply on an image in a row for RandomAugment
                        M: Magnitude of transformations for RandomAugment (0, 1, or 2)
                        lta: Whether the augmentation agent should be trained for elastic morphing (with Learn to Augment)
                        agent: Augmentation agent model
                        agent_opt: (Learning augmentation) agent optimizer
                        randaug_sampler: Used to apply RandomAugment to images
    @param labels: True classes for the train set
    @param recognizer: Recognizer/classifier model
    @param recognizer_opt: Recognizer optimizer
    @param classes: Classes (order given by the encoder)
    @param test_labels: True classes for the test set
    @param testloader: Used to loop over test data.
    @param train_recog: Whether the recognizer should be trained
    @param split: Test split percentage
    
    @return episode classifier and augmentation loss, and accuracy
    """
    # Set up
    trainset = torch.utils.data.TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    loss, aug_loss, acc = 0.0, 0.0, 0.0
    batch_progress = tqdm(trainloader, desc='Batches', leave=False)

    # Batch loop
    for i, (images, labels) in enumerate(batch_progress):
        aug_S2, agent_outputs, S, S2 = None, None, None, None

        # Augment with elastic morphing
        if "elastic" in aug_dict['type']:
            if aug_dict['lta']:
                images, aug_S2, agent_outputs, S, S2 = augmentation.augment_data(images, aug_dict['n_patches'], aug_dict['radius'], agent=aug_dict['agent'])
                aug_S2 = aug_S2.to(device)
            else:
                images = augmentation.augment_data(images, aug_dict['n_patches'], aug_dict['radius'])

        # Augment with RandomAugment
        if aug_dict['randaug_sampler'] is not None:
            images = augmentation.randaug_data(images, aug_dict['randaug_sampler'])

        images, labels = images.to(device), labels.to(device)

        # Train
        lta_dict = {'aug_S2': aug_S2,
                    'agent_outputs': agent_outputs,
                    'S': S,
                    'S2': S2,
                    'agent_opt': aug_dict['agent_opt'],
                    'lta': aug_dict['lta']}
        batch_loss, batch_aug_loss = train_batch(images, labels, lta_dict, recognizer, recognizer_opt, train_recog=train_recog)

        # Evaluate accuracy on test split
        if split:
            correct, total, _, _ = test_classifier(classes, recognizer, test_labels, testloader)
            batch_acc = correct / total
        else: batch_acc = 0.0

        loss += batch_loss; aug_loss += batch_aug_loss; acc += batch_acc
        batch_progress.set_postfix(loss=loss/(i+1), acc=acc/(i+1))
    return loss/len(trainloader), aug_loss/len(trainloader), acc/len(trainloader)

def load_and_test(batch_size, model_name, split=0):
    """
    Load and test a classier on the test set.
    @param batch_size: Batch size
    @param model_name: Filename of the model (without .pth)
    @param split: Test split percentage
    """
    classes, labels, testloader, classifier = load_test_data(batch_size, split, model_name)
    
    correct, total, correct_pred, total_pred = test_classifier(classes, classifier, labels, testloader)

    print(f'Accuracy: {100 * correct / total:.2f} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname.decode()} is {accuracy:.1f} % ({correct_count}/{total_pred[classname]})')

def test_classifier(classes, classifier, labels, testloader):
    """
    Test a classifier on the test set of segmented characters.
    @param classes: Classes (order given by the encoder)
    @param classifier: Classifier model
    @param labels: True classes of the test images
    @param testloader: Test images
    
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


def train(n_epochs, batch_size, aug_dict, train_recog=True, split=0):
    """
    Train the recognizer/classifier.
    @param n_epochs: Number of epochs to train for
    @param batch_size: Batch size
    @param aug_dict: Contains:
                        type: 'elastic' and/or 'randaug' for Elastic morphing and RandomAugment to be applied
                        n_patches: Number of patches used for elastic morphing.
                        radius: Radius used for elastic morphing.
                        N: Number of transformations to apply on an image in a row for RandomAugment
                        M: Magnitude of transformations for RandomAugment (0, 1, or 2)
                        lta: Whether the augmentation agent should be trained for elastic morphing (with Learn to Augment)
    @param train_recog: Whether the recognizer should be trained (otherwise the latest version is loaded, useful for testing Learn to Augment)
    @param split: Test split percentage
    """
    org_images, labels = load_training_data(split)

    # train on subset of data
    # idxs = np.random.randint(org_images.shape[0], size=100)
    # idxs = [0]
    # org_images = org_images[idxs]
    # labels = labels[idxs]

    aug_dict['agent'], aug_dict['agent_opt'], aug_dict['randaug_sampler'], classes, test_labels, testloader, recognizer_opt = None, None, None, None, None, None, None

    # Set up augmentation agent
    if aug_dict['lta']:
        n_points = 2*(aug_dict['n_patches']+1)
        aug_dict['agent'] = networks.AugmentAgentCNN(n_points).to(device)
        # summary(aug_dict['agent'], input_size=(1, 1, h, w), device=device)

        aug_dict['agent_opt'] = optim.Adadelta(aug_dict['agent'].parameters())

        augment_path = "augmented"
        os.makedirs(augment_path, exist_ok=True)

    # Set up RandomAugment sampler
    if "randaug" in aug_dict['type']:
        aug_dict['randaug_sampler'] = augmentation.RandomAug(aug_dict['N'], aug_dict['M'])

    # Set up recognizer/classifier and training data
    if train_recog:
        recognizer = networks.ClassifierCNN(CHARACTER_HEIGHT).to(device)
        # summary(recognizer, input_size=(1, 1, CHARACTER_HEIGHT, CHARACTER_WIDTH), device=device)

        recognizer_opt = optim.AdamW(recognizer.parameters(), amsgrad=True)
        scheduler = optim.lr_scheduler.MultiStepLR(recognizer_opt, milestones = [120, 140], gamma = 0.1)

        if split:
            classes, test_labels, testloader = load_test_data(batch_size, split)
            
    # Alternatively load recognizer/classifier and training data
    elif split:
        classes, test_labels, testloader, recognizer = load_test_data(batch_size, split, uniquify(get_model_save_name('recognizer', aug_dict), find=True))
    
    recognizer = recognizer.to(device)
    losses, aug_losses, accs = [], [], []

    # Epoch loop
    for ep in tqdm(range(n_epochs), desc='Epochs'):
        ep_loss, ep_aug_loss, ep_acc = train_epoch(org_images,
                                                    batch_size,
                                                    aug_dict,
                                                    labels,
                                                    recognizer,
                                                    recognizer_opt,
                                                    classes,
                                                    test_labels,
                                                    testloader,
                                                    train_recog=train_recog,
                                                    split=split
                                                    )
            
        if aug_dict['lta']:
            # save example augmented images from Learn to Augment agent
            ex_images = org_images[[0, 500, 1000]].to(device)
            agent_outputs = aug_dict['agent'](ex_images)
            ex_images = torch.squeeze(ex_images, 1).detach().cpu().numpy()
            S = torch.max(agent_outputs, 3).indices.detach().cpu().numpy()
            for i in range(ex_images.shape[0]):
                example_dist, src_pts, dst_pts = augmentation.distort(ex_images[i], aug_dict['n_patches'], aug_dict['radius'], S[i], return_points=True, max_radius=True)
                example = augmentation.draw_augment_arrows(ex_images[i], src_pts, dst_pts, aug_dict['radius'])
                cv2.imwrite(os.path.join(augment_path, f"{i}_e{ep}.png"), example)
                cv2.imwrite(os.path.join(augment_path, f"{i}_e{ep}_dist.png"), example_dist)

        scheduler.step()
        losses.append(ep_loss); aug_losses.append(ep_aug_loss); accs.append(ep_acc)
        print(f"\nEp {ep}. Train Loss: {ep_loss:.2f}. Test Accuracy: {ep_acc:.4f}.")

    # Save results
    if train_recog:
        save_name = save_model('recognizer', recognizer, recognizer_opt, aug_dict)
        plot_final(losses, 'Classifier Loss')
        plt.savefig(f"{save_name}_loss.png")
        plot_final(accs, 'Accuracy')
        plt.savefig(f"{save_name}_acc.png")
    if aug_dict['lta']:
        save_name = save_model('agent', aug_dict['agent'], aug_dict['agent_opt'], aug_dict)
        plot_final(aug_losses, 'Agent Loss', ylim=[0,1])
        plt.savefig(f"{save_name}_aug_loss.png")
    plt.show()