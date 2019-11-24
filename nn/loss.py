import torch
from torch import nn
import torch.nn.functional as F


class MultiViewLoss(nn.Module):
    def __init__(self):
        super(MultiViewLoss, self).__init__()

    def forward(self, f1_out, f2_out, ft_out, w1, w2, y, _del):
        y = y.to(torch.float)
        f1_ce = F.binary_cross_entropy(f1_out, y, reduction='none')
        f2_ce = F.binary_cross_entropy(f2_out, y, reduction='none')
        ft_ce = F.binary_cross_entropy(ft_out, y, reduction='none')
        diff = torch.abs(torch.sum(torch.abs(w1)) - torch.sum(torch.abs(w2)))
        w1 = w1.reshape(-1).view(2500, 1).t()
        w2 = w2.reshape(-1).view(2500, 1)
        regularizer = torch.mm(w1, w2)
        #print(regularizer.item(), diff)
        return torch.mean(f1_ce + f2_ce) + 0.001 * torch.abs(torch.mm(w1, w2)) # + 0.001 * diff
