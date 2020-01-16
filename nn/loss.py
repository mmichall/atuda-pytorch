import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MSELoss, KLDivLoss


class MultiViewLoss(nn.Module):

    def __init__(self):
        super(MultiViewLoss, self).__init__()

    def forward(self, f1_out, f2_out, w1, w2, enc_src_out, enc_tgt_out, y):
        f1_ce = F.cross_entropy(f1_out, y, reduction='none')
        f2_ce = F.cross_entropy(f2_out, y, reduction='none')
        diff = torch.abs(torch.sum(torch.abs(w1)) - torch.sum(torch.abs(w2)))
        w1 = w1.reshape(-1)
        w2 = w2.reshape(-1)
        w1 = w1.view(len(w1), 1).t()
        w2 = w2.view(len(w2), 1)

        P_prob = torch.softmax(torch.mean(enc_src_out, dim=0), 0)
        Q_prob = torch.softmax(torch.mean(enc_tgt_out, dim=0), 0)

        d = (KL(P_prob, Q_prob) + KL(Q_prob, P_prob)) / 2

        # regularizer = torch.mm(w1, w2)
        # print(regularizer.item())
        # print(d.item())
        return torch.mean(f1_ce + f2_ce) + torch.abs(torch.mm(w1, w2)) + 0.001 * diff + d


class KLDivergenceLoss(nn.Module):

    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, p, q):
        return KL(p, q)


class MSEWithDivergenceLoss(nn.Module):
    def __init__(self):
        super(MSEWithDivergenceLoss, self).__init__()
        self.alpha = 1
        self.beta = 1
        self.reconstruction_cr = MSELoss()
        self.divergence_cr = KLDivLoss()

    def forward(self, p, q, tgt_p, tgt_q, hidden_p, hidden_q):
        r = self.reconstruction_cr(p, q)
        r_tgt = self.reconstruction_cr(tgt_p, tgt_q)

        # P_prob = torch.softmax(torch.mean(hidden_p, dim=0), 0)
        # Q_prob = torch.softmax(torch.mean(hidden_q, dim=0), 0)

        # d = (KL(P_prob, Q_prob) + KL(Q_prob, P_prob)) / 2
        # print(' ' + str(d.item()), str(r.item()))
        return (r + r_tgt) / 2 # + d


class ReversalLoss(nn.Module):
    def __init__(self):
        super(ReversalLoss, self).__init__()

    def forward(self, out, rev_out, y, rev_y):
        rev_out = torch.squeeze(rev_out)
        rev_y = torch.squeeze(rev_y)
        mse_err = F.mse_loss(out, y, reduction='mean')
        domain_err = F.binary_cross_entropy_with_logits(rev_out, rev_y, reduction='mean')

        return domain_err


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    P = torch.softmax(torch.mean(P, dim=0), 0)
    Q = torch.softmax(torch.mean(Q, dim=0), 0)

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    d = (P * (P / Q).log()).sum()
    return d
