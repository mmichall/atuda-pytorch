import torch
from torch import Tensor
from torch.autograd import Variable


def acc(output: Tensor, ground_truth: Tensor):
    t = Variable(torch.FloatTensor([0.5]))  # threshold
    out1 = (output > t)
    out = out1.cpu().numpy().flatten()
    labels = ground_truth.cpu().numpy()

    return (out == labels).sum() / len(labels)