from ensemble import Ensemble
from cnn import ConvNet
from train_seq import train_sequence
from binet import BiNet
from biloss import BiLoss
from torch.optim import Adam
from sequence_loader import get_sequence_loader
import torchvision.transforms as tf
from accuracy import Accuracy


def run():
    ensemble = Ensemble(
        model=ConvNet,
        size=5,
        img_size=32,
        num_classes=27,
    )

    loader = get_sequence_loader(
        bigram_data='../data/bigrams.csv',
        image_dir='../data/monkbrill/',
        length=5000,
        transform=tf.Resize((32, 32)),
        batch_size=64,
    )

    train_sequence(
        model=BiNet(27),
        ensemble=ensemble,
        loss_function=BiLoss(),
        criterion=Accuracy(),
        epochs=1000,
        optimizer_class=Adam,
        loader=loader
    )


if __name__ == '__main__':
    run()

