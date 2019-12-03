import random
import sys

import config
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
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
    data_set.denoising_factor = 0.5

    data_set.summary('data_set')

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}

    max_epochs = 15

    # Generators
    data_set_generator = DataLoader(data_set, **params)

    model = SimpleAutoencoder(5000, 500)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = MSELoss()

    epochs_no_improve = 3
    n_epochs_stop = 0

   # model.embedding_mode()




