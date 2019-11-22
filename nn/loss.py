import torch
from torch import nn
import torch.nn.functional as F


class MultiViewLoss(nn.Module):
    def __init__(self):
        super(MultiViewLoss, self).__init__()

    def forward(self, f1_out, f2_out, w1, w2, y, _del):
        y = y.to(torch.long)
        f1_ce = F.cross_entropy(f1_out, y, reduction='none')
        f2_ce = F.cross_entropy(f2_out, y, reduction='none')
        w1 = w1.reshape(-1).view(2500, 1).t()
        w2 = w2.reshape(-1).view(2500, 1)
        regularizer = torch.mm(w1, w2)
        return torch.mean(f1_ce + f2_ce) + 0.0001 * regularizer
