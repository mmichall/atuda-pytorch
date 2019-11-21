from torch.autograd import Variable

import config
import torch
import sys
from torch.utils.data import DataLoader
from model import Feedforward
from data_set import SentimentDataSet, MultiDomainSentimentDataSet
from pprint import pprint
import numpy as np

if __name__ == '__main__':
    # CUDA for PyTorch
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)

    multidataset = MultiDomainSentimentDataSet(config.SOURCE_DOMAIN, config.TARGET_DOMAIN, n=5000)

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}

    params_valid = {'batch_size': len(multidataset.tgt_ds)}
    max_epochs = 100

    # Generators
    training_generator = DataLoader(multidataset.src_ds, **params)
    validation_generator = DataLoader(multidataset.tgt_ds, **params_valid)

    model = Feedforward(5000, 10)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        batch = 0
        loss_all = 0
        for local_batch, local_labels in training_generator:
            batch += 1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device, torch.float), local_labels.to(device, torch.float)

            # Model computations
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(local_batch)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), local_labels)
            loss_all += loss.item()

            t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
            out = (y_pred > t)
            out = out.cpu().numpy().flatten()
            local_labels = local_labels.cpu().numpy()

            acc = (out == local_labels).sum() / len(local_labels)

            sys.stdout.write('\rEpoch {}, Batch {}, train loss: {}, acc: {}'.format(epoch, batch, round(loss_all / batch, 4), round(acc * 100, 2)))
            sys.stdout.flush()
            # Backward pass
            loss.backward()
            optimizer.step()
        sys.stdout.write('\n')

        # Validation
        acc = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch = local_batch.to(device, torch.float)

                # Model computations
                y_pred = model(local_batch)
                t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
                out = (y_pred > t)
                out = out.cpu().numpy().flatten()
                local_labels = local_labels.cpu().numpy()

                acc = (out == local_labels).sum() / len(local_labels)

                print('\rValidation accuracy: {}%'.format(round(acc * 100, 2)))
