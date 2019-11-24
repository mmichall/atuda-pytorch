import torch


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
       # self.f_softmax = torch.nn.Softmax()

        self.f1_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f1_1_relu = torch.nn.ReLU()
        self.f1_2 = torch.nn.Linear(self.hidden_size, 1)
        self.f1_2_softmax = torch.nn.Sigmoid()

        self.f2_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f2_1_relu = torch.nn.ReLU()
        self.f2_2 = torch.nn.Linear(self.hidden_size, 1)
        self.f2_2_softmax = torch.nn.Sigmoid()

        self.f3_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f3_1_relu = torch.nn.ReLU()
        self.f3_2 = torch.nn.Linear(self.hidden_size, 1)
        self.f3_2_softmax = torch.nn.Sigmoid()

        self.ft = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.ft_relu = torch.nn.ReLU()
        self.ft_softmax = torch.nn.Softmax()

    def forward(self, x):
        f_hidden = self.f(x)
        f_relu = self.f_relu(f_hidden)

        # output = self.f1_1(f_relu)
        # output = self.f1_1_relu(output)
        output = self.f1_2(f_relu)
        output1 = self.f1_2_softmax(output)
        #F1_softmax = self.f1_softmax(F1_relu)

        output = self.f2_1(f_relu)
        output = self.f2_1_relu(output)
        output = self.f2_2(output)
        output2 = self.f2_2_softmax(output)

        output = self.f3_1(f_relu)
        output = self.f3_1_relu(output)
        output = self.f3_2(output)
        output3 = self.f3_2_softmax(output)

        # ft_hidden = self.ft(f_relu)
        # ft_relu = self.ft_relu(ft_hidden)
        # #Ft_softmax = self.f1_softmax(Ft_relu)

        return output1, output2, output3


