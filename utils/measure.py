import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def acc(output: Tensor, ground_truth: Tensor):
    t = Variable(torch.FloatTensor([0.5]).to(device))
    _ground_truth = ground_truth.to(dtype=torch.uint8).cpu().numpy()
    _out = (output > t).cpu().numpy()
    sum_ = 0
    for i in range(0, len(_ground_truth)):
        if np.array_equal(_ground_truth[i], _out[i]):
            sum_ += 1
    return sum_ / len(_ground_truth)