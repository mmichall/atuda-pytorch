import torch
import numpy as np
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class AE_Generator(torch.nn.Module):

    def __init__(self, input_size, latent_dim):
        super(AE_Generator, self).__init__()

        def block(in_feat, out_feat, dropout_p=None, normalize=False, return_logits=False):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            if not return_logits:
                layers.append(torch.nn.ReLU())
            if dropout_p is not None:
                layers.append(torch.nn.Dropout(p=dropout_p))
            return layers

        self.encoder = torch.nn.Sequential(
            *block(input_size, 500),
            *block(500, latent_dim, return_logits=True)
        )

        self.decoder = torch.nn.Sequential(
            *block(latent_dim, 500),
            *block(500, input_size, return_logits=True)
        )

        self.domain_discriminator = torch.nn.Sequential(
            *block(latent_dim, 100, dropout_p=0.5),
            *block(100, 1, return_logits=True)
        )

        self.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.relu(encoded)
        decoded = self.decoder(encoded)
        domain = self.domain_discriminator(encoded)
        return decoded, domain

    def summary(self):
        print('> Model summary: \n{}'.format(self))

