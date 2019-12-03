import random

import torch
from torch.nn import BatchNorm1d
from torch import nn


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class ATTFeedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ATTFeedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f = torch.nn.Linear(self.input_size, self.hidden_size)
        self.f_relu = torch.nn.ReLU()
        self.f_dropout = torch.nn.Dropout(p=0.5)
        #self.f_batchnorm = torch.nn.BatchNorm1d(50)
        # self.f_softmax = torch.nn.Softmax()

        self.f1_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f1_1_relu = torch.nn.ReLU()
        self.f1_dropout = torch.nn.Dropout(p=0.3)
        self.f1_batchnorm = BatchNorm1d(50)
        self.f1_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f1_2_softmax = torch.nn.Softmax()

        self.f2_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f2_1_relu = torch.nn.ReLU()
        self.f2_dropout = torch.nn.Dropout(p=0.3)
        self.f2_batchnorm = BatchNorm1d(50)
        self.f2_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f2_2_softmax = torch.nn.Softmax()

        self.f3_1 = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.f3_1_relu = torch.nn.ReLU()
        self.f3_dropout = torch.nn.Dropout(p=0.3)
        self.f3_batchnorm = BatchNorm1d(50)
        self.f3_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f3_2_softmax = torch.nn.Softmax()

    def forward(self, x):
        output = self.f(x)
        output = self.f_relu(output)
        f_relu = self.f_dropout(output)
       # f_relu = self.f_batchnorm(f_relu)

        output1 = self.f1_1(f_relu)
        output1 = self.f1_1_relu(output1)
        output1 = self.f1_dropout(output1)
        output1 = self.f1_2(output1)
        output1 = self.f1_2_softmax(output1)
        #F1_softmax = self.f1_softmax(F1_relu)

        output2 = self.f2_1(f_relu)
        output2 = self.f2_1_relu(output2)
        output2 = self.f2_dropout(output2)
        output2 = self.f2_2(output2)
        output2 = self.f2_2_softmax(output2)

        output3 = self.f3_1(f_relu)
        output3 = self.f3_1_relu(output3)
        output3 = self.f3_dropout(output3)
        output3 = self.f3_2(output3)
        output3 = self.f3_2_softmax(output3)

        # ft_hidden = self.ft(f_relu)
        # ft_relu = self.ft_relu(ft_hidden)
        # #Ft_softmax = self.f1_softmax(Ft_relu)

        return output1, output2, output3


class AE_ATTFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AE_ATTFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        torch.load('tmp/model.pkl')
        self.endocer_state_dict = torch.load('tmp/ae_model.pkl')
        for item in self.endocer_state_dict.items():
            print(item)
        pretrained_dict = {k: v for k, v in self.endocer_state_dict.items() if k in []}


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.train_mode = True

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,  self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size,  250),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(250, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid())

    def train_mode(self):
        self.train_mode = True

    def embedding_mode(self):
        self.train_mode = False

    def forward(self, x):
        x = self.encoder(x)
        if self.train_mode:
            x = self.decoder(x)
        return x



