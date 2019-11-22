import sys

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)


class DomainAdaptationTrainer:

    def __init__(self, model, criterion, optimizer, max_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.model.to(device)

    def fit(self, training_generator, validation_generator, tgt_domain_validation_generator):
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
                f1, f2, w1, w2, ft = self.model(local_batch)
                # Compute Loss
                # f1_out, f2_out, w1, w2, y, _del
                loss = self.criterion(f1, f2, w1, w2, local_labels, 0.01)
                loss_all += loss.item()

                # t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
                # out = (y_pred > t)
                # out = out.cpu().numpy().flatten()
                # local_labels = local_labels.cpu().numpy()

                # acc = (out == local_labels).sum() / len(local_labels)

                sys.stdout.write('\r### Epoch {}, Batch {}, train loss: {} ###'.format(epoch, batch, loss.item()))
                sys.stdout.flush()
                # Backward pass
                loss.backward()
                self.optimizer.step()

            sys.stdout.write('\n')

            # # Validation
            # acc = 0
            # with torch.set_grad_enabled(False):
            #     for local_batch, local_labels in validation_generator:
            #         # Transfer to GPU
            #         local_batch = local_batch.to(device, torch.float)
            #
            #         # Model computations
            #         y_pred = self.model(local_batch)
            #         t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
            #         out = (y_pred > t)
            #         out = out.cpu().numpy().flatten()
            #         local_labels = local_labels.cpu().numpy()
            #
            #         acc = (out == local_labels).sum() / len(local_labels)
            #         acc = round(acc * 100, 2)
            #
            #     for local_batch, local_labels in tgt_domain_validation_generator:
            #         # Transfer to GPU
            #         local_batch = local_batch.to(device, torch.float)
            #
            #         # Model computations
            #         y_pred = self.model(local_batch)
            #         t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
            #         out = (y_pred > t)
            #         out = out.cpu().numpy().flatten()
            #         local_labels = local_labels.cpu().numpy()
            #
            #         acc_tgt_dmn = (out == local_labels).sum() / len(local_labels)
            #         acc_tgt_dmn = round(acc_tgt_dmn * 100, 2)
            #
            #         print('\r src_dmn_val_acc: {}% \n tgt_dmn_val_acc: {}%\n'.format(acc, acc_tgt_dmn))
