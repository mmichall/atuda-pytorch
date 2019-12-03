import random
from torchsummary import summary
import torch
from torch.nn import BatchNorm1d
from torch import nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


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

        self.to(device)

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


class StackedAutoEncoder(nn.Module):
    def __init__(self, shape: tuple):
        super(StackedAutoEncoder, self).__init__()
        self.train_mode = True

        self.encoder_modules = []
        self.decoder_modules = []
        for _i in range(len(shape)-1):
            self.encoder_modules.append(nn.Linear(shape[_i], shape[_i+1]))
            self.encoder_modules.append(nn.ReLU(True))

        for _i in reversed(range(len(shape)-1)):
            self.decoder_modules.append(nn.Linear(shape[_i+1], shape[_i]))
            if _i != 0:
                self.decoder_modules.append(nn.ReLU(True))
            else:
                self.decoder_modules.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*self.encoder_modules)
        self.decoder = nn.Sequential(*self.decoder_modules)

        self.to(device)

        print('> AutoEncoder summary: ')
        summary(self, input_size=(shape[0], ))

    def train_mode(self):
        self.train_mode = True

    def frozen(self):
        self.train_mode = False

    def forward(self, x):
        x = self.encoder(x)
        if self.train_mode:
            x = self.decoder(x)
        return x



