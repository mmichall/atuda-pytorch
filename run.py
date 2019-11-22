from torch.autograd import Variable

import config
import torch
import sys
from torch.utils.data import DataLoader
from nn.model import Feedforward
from data_set import AmazonDomainDataSet, AmazonDomainSubset
from helper.data import train_valid_split, build_dictionary
from nn.trainer import Trainer

if __name__ == '__main__':

    src_domain_data_set = AmazonDomainDataSet(config.SOURCE_DOMAIN)
    tgt_domain_data_set = AmazonDomainDataSet(config.TARGET_DOMAIN)

    dictionary = build_dictionary([src_domain_data_set, tgt_domain_data_set], 5000)

    src_domain_data_set.dict = dictionary
    tgt_domain_data_set.dict = dictionary

    src_domain_data_set.summary('src_domain_data_set')
    tgt_domain_data_set.summary('tgt_domain_data_set')

    train_idxs, valid_idxs = train_valid_split(0, len(src_domain_data_set), 0.2)

    print("Training set length: {}, Validation set length: {}".format(len(train_idxs), len(valid_idxs)))

    train_subset = AmazonDomainSubset(src_domain_data_set, train_idxs)
    valid_subset = AmazonDomainSubset(src_domain_data_set, valid_idxs)

    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}

    params_valid = {'batch_size': len(valid_subset)}
    max_epochs = 100

    # Generators
    training_generator = DataLoader(train_subset, **params)
    validation_generator = DataLoader(valid_subset, **params_valid)

    model = Feedforward(5000, 100)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(model, criterion, optimizer, max_epochs)
    trainer.fit(training_generator, validation_generator)
