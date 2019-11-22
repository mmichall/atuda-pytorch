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

        self.f1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f1_relu = torch.nn.ReLU()
        self.f1_softmax = torch.nn.Softmax()

        self.f2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f2_relu = torch.nn.ReLU()
        self.f2_softmax = torch.nn.Softmax()

        self.ft = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.ft_relu = torch.nn.ReLU()
        self.ft_softmax = torch.nn.Softmax()

    def forward(self, x):
        f_hidden = self.f(x)
        f_relu = self.f_relu(f_hidden)

        f1_hidden = self.f1(f_relu)
        f1_relu = self.f1_relu(f1_hidden)
        #F1_softmax = self.f1_softmax(F1_relu)

        f2_hidden = self.f2(f_relu)
        f2_relu = self.f2_relu(f2_hidden)
        #F2_softmax = self.f1_softmax(F2_relu)

        ft_hidden = self.ft(f_relu)
        ft_relu = self.ft_relu(ft_hidden)
        #Ft_softmax = self.f1_softmax(Ft_relu)

        return f1_relu, f2_relu, ft_relu


