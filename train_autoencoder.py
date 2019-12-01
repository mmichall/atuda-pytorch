import sys

import config
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from nn.model import SimpleAutoencoder
from data_set import AmazonDomainDataSet, AmazonSubsetWrapper
from helper.data import train_valid_split, build_dictionary
from helper.dataset import merge

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if __name__ == '__main__':

    print('SOURCE domain: {}'.format(config.SOURCE_DOMAIN))
    print('TARGET domain: {}'.format(config.TARGET_DOMAIN))

    src_domain_data_set = AmazonDomainDataSet(config.SOURCE_DOMAIN, True)
    tgt_domain_data_set = AmazonDomainDataSet(config.TARGET_DOMAIN, False)

    dictionary = build_dictionary([src_domain_data_set, tgt_domain_data_set], 5000)

    src_domain_data_set.dict = dictionary
    tgt_domain_data_set.dict = dictionary

    data_set = merge([src_domain_data_set, tgt_domain_data_set])
    data_set.dict = dictionary

    data_set.summary('data_set')

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}

    max_epochs = 100

    # Generators
    training_generator = DataLoader(src_domain_data_set, **params)
    data_set_generator = DataLoader(data_set, **params)

    model = SimpleAutoencoder(5000, 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion = MSELoss()

    epochs_no_improve = 3
    n_epochs_stop = 0

    model.train()
    for epoch in range(max_epochs):
        _loss = 0
        _batch = 0
        prev_loss = 999
        print('Epoch: ' + str(epoch))
        for idx, batch_one_hot, labels, src in data_set_generator:
            _batch += 1
            labels = torch.stack(labels, dim=1)
            batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)
            optimizer.zero_grad()

            out = model(batch_one_hot.cpu())

            loss = criterion(out, batch_one_hot)

            _loss += loss.item()
            _loss_mean = round(_loss / _batch, 4)
            sys.stdout.write('\r MSE Error: ' + str(_loss_mean))
            loss.backward()
            optimizer.step()
        if prev_loss <= _loss_mean:
            n_epochs_stop += 1
        else:
            n_epochs_stop = 0
            prev_loss = _loss_mean

        if n_epochs_stop == epochs_no_improve:
            print('Early Stopping!')
            break

        print('')
    torch.save(model.state_dict(), 'tmp/ae_model.pkl')


