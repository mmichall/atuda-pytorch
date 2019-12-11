import torch
from torch import nn
import torch.nn.functional as F


class MultiViewLoss(nn.Module):
    def __init__(self):
        super(MultiViewLoss, self).__init__()

    def forward(self, f1_out, f2_out, w1, w2, y):
        f1_ce = F.cross_entropy(f1_out, y, reduction='none')
        f2_ce = F.cross_entropy(f2_out, y, reduction='none')
        # diff = torch.abs(torch.sum(torch.abs(w1)) - torch.sum(torch.abs(w2)))
        w1 = w1.reshape(-1)
        w2 = w2.reshape(-1)
        w1 = w1.view(len(w1), 1).t()
        w2 = w2.view(len(w2), 1)
        # regularizer = torch.mm(w1, w2)
        # print(regularizer.item())
        return torch.mean(f1_ce + f2_ce) + 0.001 * torch.abs(torch.mm(w1, w2)) # + 0.001 * diff


class ReversalLoss(nn.Module):
    def __init__(self):
        super(ReversalLoss, self).__init__()

    def forward(self, out, rev_out, y, rev_y):
        rev_out = torch.squeeze(rev_out)
        rev_y = torch.squeeze(rev_y)
        mse_err = F.mse_loss(out, y, reduction='mean')
        domain_err = F.binary_cross_entropy_with_logits(rev_out, rev_y, reduction='mean')

        return domain_err
