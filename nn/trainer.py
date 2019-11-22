import sys

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)


class Trainer:

    def __init__(self, model, criterion, optimizer, max_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.model.to(device)

    def fit(self, training_generator, validation_generator):
        # Loop over epochs
        for epoch in range(self.max_epochs):
            # Training
            batch = 0
            loss_all = 0
            for local_batch, local_labels in training_generator:
                batch += 1
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device, torch.float), local_labels.to(device, torch.float)

                # Model computations
                self.optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(local_batch)
                # Compute Loss
                loss = self.criterion(y_pred.squeeze(), local_labels)
                loss_all += loss.item()

                t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
                out = (y_pred > t)
                out = out.cpu().numpy().flatten()
                local_labels = local_labels.cpu().numpy()

                acc = (out == local_labels).sum() / len(local_labels)

                sys.stdout.write('\rEpoch {}, Batch {}, train loss: {}'.format(epoch, batch, round(loss_all / batch, 4)))
                sys.stdout.flush()
                # Backward pass
                loss.backward()
                self.optimizer.step()

            sys.stdout.write('\n')

            # Validation
            acc = 0
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    # Transfer to GPU
                    local_batch = local_batch.to(device, torch.float)

                    # Model computations
                    y_pred = self.model(local_batch)
                    t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
                    out = (y_pred > t)
                    out = out.cpu().numpy().flatten()
                    local_labels = local_labels.cpu().numpy()

                    acc = (out == local_labels).sum() / len(local_labels)

                    print('\rValidation accuracy: {}%'.format(round(acc * 100, 2)))
