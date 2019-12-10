import random
from torchsummary import summary
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch import nn, optim

from nn.RevGrad import RevGrad

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class ATTFeedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ATTFeedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f = torch.nn.Linear(self.input_size, self.hidden_size)
        self.f_relu = torch.nn.ReLU()
        self.f_dropout = torch.nn.Dropout(p=0.3)
        self.f_batchnorm = torch.nn.BatchNorm1d(50)
        # self.f_softmax = torch.nn.Softmax()

        self.f1_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f1_1_relu = torch.nn.ReLU()
        self.f1_dropout = torch.nn.Dropout(p=0.2)
        self.f1_batchnorm = BatchNorm1d(50)
        self.f1_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f1_2_softmax = torch.nn.Softmax(dim=0)

        self.f2_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f2_1_relu = torch.nn.ReLU()
        self.f2_dropout = torch.nn.Dropout(p=0.2)
        self.f2_batchnorm = BatchNorm1d(50)
        self.f2_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f2_2_softmax = torch.nn.Softmax(dim=0)

        self.f3_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.f3_1_relu = torch.nn.ReLU()
        self.f3_dropout = torch.nn.Dropout(p=0.2)
        self.f3_batchnorm = BatchNorm1d(50)
        self.f3_2 = torch.nn.Linear(self.hidden_size, 2)
        self.f3_2_softmax = torch.nn.Softmax(dim=0)

        self.to(device)

    def forward(self, x):
        output = self.f(x)
    #    output = self.f_batchnorm(output)
        output = self.f_relu(output)
        f_relu = self.f_dropout(output)


        output1 = self.f1_1(f_relu)
 #       output1 = self.f1_batchnorm(output1)
        output1 = self.f1_1_relu(output1)
        output1 = self.f1_dropout(output1)
        output1 = self.f1_2(output1)
        #output1 = self.f1_2_softmax(output1)
#        output1 = F.softmax(output1, dim=0)
        #F1_softmax = self.f1_softmax(F1_relu)

        output2 = self.f2_1(f_relu)
    #    output2 = self.f2_batchnorm(output2)
        output2 = self.f2_1_relu(output2)
        output2 = self.f2_dropout(output2)
        output2 = self.f2_2(output2)

        #output2 = self.f2_2_softmax(output2)

        output3 = self.f3_1(f_relu)
    #    output3 = self.f3_batchnorm(output3)
        output3 = self.f3_1_relu(output3)
        output3 = self.f3_dropout(output3)
        output3 = self.f3_2(output3)

     #   output3 = self.f3_2_softmax(output3)

        # ft_hidden = self.ft(f_relu)
        # ft_relu = self.ft_relu(ft_hidden)
        # #Ft_softmax = self.f1_softmax(Ft_relu)

        return output1, output2, output3

    def get_f1_1(self):
        return self.f1_1

    def get_f2_1(self):
        return self.f2_1

    def summary(self):
        print('> Model summary: \n{}'.format(self))
        summary(self, input_size=(self.hidden_size,  self.input_size, ))


class StackedAutoEncoder(nn.Module):
    def __init__(self, shape: tuple):
        super(StackedAutoEncoder, self).__init__()
        self.train_mode = True
        self.shape = shape

        self.reversal_layer = RevGrad()
        self.domain_classifier = nn.Linear(250, 1)
        self.reversal = nn.Sequential(self.reversal_layer, self.domain_classifier)

        self.encoder_modules = []
        self.decoder_modules = []
        for _i in range(len(shape)-1):
            seq_list = []
            seq_list.append(nn.Linear(shape[_i], shape[_i+1]))
            seq_list.append(nn.ReLU(True))
            self.encoder_modules.append(nn.Sequential(*seq_list))

        for _i in reversed(range(len(shape)-1)):
            seq_list = []
            seq_list.append(nn.Linear(shape[_i+1], shape[_i]))
            if _i != 0:
                seq_list.append(nn.ReLU(True))
            else:
                seq_list.append(nn.Sigmoid())
            self.decoder_modules.append(nn.Sequential(*seq_list))

        self.encoder = nn.Sequential(*self.encoder_modules)
        self.decoder = nn.Sequential(*self.decoder_modules)

        self.to(device)

    def train_mode(self):
        self.train()
        self.train_mode = True

    def froze(self):
        self.eval()
        self.train_mode = False

    def forward(self, x):
        result = torch.empty(0).cuda()
        if not self.train_mode:
            for layer in self.encoder:
                x = layer(x)
                result = torch.cat((result, x), 1)
            return result
        else:
            x = self.encoder(x)
            domain_out = self.reversal(x)
            y = self.decoder(x)
        return y, domain_out

    def summary(self):
        print('> Model summary: \n{}'.format(self))
        summary(self, input_size=(self.shape[0], ))


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, ae_model=None):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.ae_model = ae_model
        self.temperatures = [nn.Parameter(torch.ones(1) * 1.5), nn.Parameter(torch.ones(1) * 1.5)]

    def forward(self, input):
        f1_logits, f2_logits, fn_logits = self.model(input)
        return self.temperature_scale(f1_logits, self.temperatures[0]),\
               self.temperature_scale(f2_logits, self.temperatures[1]),\
               fn_logits
        # return self.model(input)

    def temperature_scale(self, logits, temperature):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        f1_logits_list = []
        f2_logits_list = []
        fn_logits_list = []
        labels_list = []
        with torch.no_grad():
            for idx, input, label, src in valid_loader:
                input = input.to(device, dtype=torch.float)
                if self.ae_model:
                    ae_output = self.ae_model(input)
                    input = torch.cat([input, ae_output], 1)
                f1_logits, f2_logits, fn_logits = self.model(input)
                f1_logits_list.append(f1_logits)
                f2_logits_list.append(f2_logits)
                #fn_logits_list.append(fn_logits)
                labels_list.append(torch.stack(label, dim=1))
            f1_logits = torch.cat(f1_logits_list).cuda()
            f2_logits = torch.cat(f2_logits_list).cuda()
            #fn_logits = torch.cat(fn_logits_list).cuda()

            labels = torch.cat(labels_list)
            labels = torch.max(labels, dim=1)[1].cuda()

        # Calculate NLL and ECE before temperature scaling
        for i, logits in enumerate((f1_logits, f2_logits)):
            before_temperature_nll = nll_criterion(logits, labels).item()
            print('before')
            before_temperature_ece = ece_criterion(logits, labels).item()
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperatures[i]], lr=0.01, max_iter=50)

            def eval():
                loss = nll_criterion(self.temperature_scale(logits, self.temperatures[i]), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits, self.temperatures[i]), labels).item()
            print('after')
            after_temperature_ece = ece_criterion(self.temperature_scale(logits, self.temperatures[i]), labels).item()
            print('Optimal temperature: %.3f' % self.temperatures[i].item())
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    def get_f1_1(self):
        return self.model.f1_1

    def get_f2_1(self):
        return self.model.f2_1

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)

        accuracy_in_bins = []
        avg_confidence_in_bins = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                accuracy_in_bins.append(accuracy_in_bin.item())
                avg_confidence_in_bins.append(avg_confidence_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
       # print(accuracy_in_bins)
      #  print(avg_confidence_in_bins)

        return ece
