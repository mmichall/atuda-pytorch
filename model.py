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

        self.F = torch.nn.Linear(self.input_size, self.hidden_size)
        self.F_relu = torch.nn.ReLU()
        self.F_softmax = torch.nn.Softmax()

        self.F1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.F1_relu = torch.nn.ReLU()
        self.F1_softmax = torch.nn.Softmax()

        self.F2 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.F2_relu = torch.nn.ReLU()
        self.F2_softmax = torch.nn.Softmax()

        self.Ft = torch.nn.Linear(self.input_size, self.hidden_size)
        self.Ft_relu = torch.nn.ReLU()
        self.Ft_softmax = torch.nn.Softmax()

    def forward(self, x):
        F_hidden = self.F(x)
        F_relu = self.F_relu(F_hidden)

        F1_hidden = self.fc2(F_relu)
        F1_relu = self.F1_relu(F1_hidden)
        F1_softmax = self.F1_softmax(F1_relu)

        F2_hidden = self.fc2(F_relu)
        F2_relu = self.F1_relu(F2_hidden)
        F2_softmax = self.F1_softmax(F2_relu)

        Ft_hidden = self.fc2(F_relu)
        Ft_relu = self.F1_relu(Ft_hidden)
        Ft_softmax = self.F1_softmax(Ft_relu)


