import torch
import numpy as np
import torch.nn.functional as F


class AE_Generator(torch.nn.Module):
    def __init__(self, input_size, latent_dim):
        super(AE_Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False, return_logits=False):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            if return_logits:
                layers.append(torch.nn.ReLU(inplace=True))
            return layers

        self.encoder = torch.nn.Sequential(
            *block(input_size, 1000),
            *block(1000, 500),
            *block(500, latent_dim, return_logits=True))

        self.decoder = torch.nn.Sequential(
            *block(latent_dim, 500),
            *block(500, 1000),
            *block(1000, input_size, return_logits=True))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(F.relu(encoded))
        return decoded


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


adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()
