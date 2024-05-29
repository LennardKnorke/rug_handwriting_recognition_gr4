import torch
import torch.nn as nn
from train import train
from dataloader import get_dataloaders
from accuracy import Accuracy


class Ensemble(nn.Module):
    def __init__(self, model, size, img_size, num_classes):
        super(Ensemble, self).__init__()
        self.model = model
        self.size = size
        self.num_classes = num_classes

        self.models = []
        train_loader, val_loader = get_dataloaders(
            data_path='../data/monkbrill/',
            batch_size=64,
        )
        for _ in range(size):
            model = self.model(img_size, num_classes)
            model = train(
                model,
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss(),
                accuracy=Accuracy(),
                learning_rate=1e-3,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
            )
            self.models.append(model)

    def forward(self, x):
        preds = torch.zeros((x.shape[0], x.shape[1], self.size, self.num_classes),
                            device=x.device)
        for im_idx in range(x.shape[1]):
            for idx, model in enumerate(self.models):
                img = x[:, im_idx]
                with torch.no_grad():
                    preds[:, im_idx, idx, :] = model(img)

        final = torch.zeros((x.shape[0], x.shape[1], self.num_classes, 2),
                            device=x.device)

        final[:, :, :, 0] = torch.mean(preds, dim=-2)
        final[:, :, :, 1] = torch.std(preds, dim=-2)

        return final




