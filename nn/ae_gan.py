import torch
import numpy as np
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class AE_Generator(torch.nn.Module):

    def __init__(self, input_size, latent_dim):
        super(AE_Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False, return_logits=False):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            if not return_logits:
                layers.append(torch.nn.ReLU(inplace=True))
            #    layers.append(torch.nn.Dropout(p=0.5))
            return layers

        # self.encoder = torch.nn.Sequential(
        #     *block(input_size, 1000),
        #     *block(1000, latent_dim, return_logits=True)
        # )
        #
        # self.decoder = torch.nn.Sequential(
        #     *block(latent_dim, 1000),
        #     *block(1000, input_size, return_logits=True)
        # )
        #
        # self.label_classifier = torch.nn.Sequential(
        #     *block(latent_dim, 50),
        #     *block(50, 1, return_logits=True)
        # )
        shape = [5000, 500, 250]
        self.encoder_modules = []
        self.decoder_modules = []
        for _i in range(len(shape) - 1):
            seq_list = []
            seq_list.append(torch.nn.Linear(shape[_i], shape[_i + 1]))
            if _i != len(shape) - 2:
                seq_list.append(torch.nn.ReLU(True))
            #  seq_list.append(nn.Dropout(p=0.3))
            self.encoder_modules.append(torch.nn.Sequential(*seq_list))

        for _i in reversed(range(len(shape)-1)):
            seq_list = []
            seq_list.append(torch.nn.Linear(shape[_i+1], shape[_i]))
            if _i != 0:
                seq_list.append(torch.nn.ReLU(True))
                #seq_list.append(nn.Dropout(p=0.3))
            #else:
                #seq_list.append(nn.Sigmoid())
            self.decoder_modules.append(torch.nn.Sequential(*seq_list))

        #self.decoder_modules.append(nn.Sequential(nn.Linear(250, 5000)))

        self.encoder = torch.nn.Sequential(*self.encoder_modules)
        self.decoder = torch.nn.Sequential(*self.decoder_modules)

        self.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(F.relu(encoded))
#        y_label = self.label_classifier(F.relu(encoded))
        return decoded, 1

    def summary(self):
        print('> Model summary: \n{}'.format(self))


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(np.prod(img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

