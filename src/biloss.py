import torch.nn as nn


class BiLoss(nn.Module):
    def __init__(self, reg_coeff=0.1):
        super(BiLoss, self).__init__()
        self.reg_coeff = reg_coeff
        self.ce_loss = nn.CrossEntropyLoss()
        self.binary_ce_loss = nn.BCELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self,  prediction, target, end, dist):
        cls, end_pred = prediction
        cross_ent = self.ce_loss(cls, target)
        reg_loss = self.reg_coeff * self.kl_loss(cls, dist)
        class_loss = cross_ent + reg_loss

        bin_loss = self.binary_ce_loss(end_pred, end)

        return class_loss + bin_loss

