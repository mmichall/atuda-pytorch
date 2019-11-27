from torch.autograd import Variable
from torch.nn.modules.loss import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
import torch
import sys
from torch.utils.data import DataLoader

from nn.loss import MultiViewLoss
from nn.model import Feedforward, ATTFeedforward
from data_set import AmazonDomainDataSet, AmazonSubsetWrapper
from helper.data import train_valid_split, build_dictionary
from nn.trainer import DomainAdaptationTrainer

if __name__ == '__main__':

    print('SOURCE domain: {}'.format(config.SOURCE_DOMAIN))
    print('TARGET domain: {}'.format(config.TARGET_DOMAIN))

    src_domain_data_set = AmazonDomainDataSet(config.SOURCE_DOMAIN, True)
    tgt_domain_data_set = AmazonDomainDataSet(config.TARGET_DOMAIN, False)

    dictionary = build_dictionary([src_domain_data_set, tgt_domain_data_set], 5000)

    src_domain_data_set.dict = dictionary
    tgt_domain_data_set.dict = dictionary

    src_domain_data_set.summary('src_domain_data_set')
    tgt_domain_data_set.summary('tgt_domain_data_set')

    train_idxs, valid_idxs = train_valid_split(0, len(src_domain_data_set), 0.2)

    print("Training set length: {}, Validation set length: {}".format(len(train_idxs), len(valid_idxs)))

    train_subset = AmazonSubsetWrapper(src_domain_data_set, train_idxs)
    valid_subset = AmazonSubsetWrapper(src_domain_data_set, valid_idxs)

    params = {'batch_size': 8,
              'shuffle': True,
              'num_workers': 4}
    params_valid = {'batch_size': len(valid_subset)}

    max_epochs = 100

    # Generators
    training_generator = DataLoader(train_subset, **params)
    validation_generator = DataLoader(valid_subset, **params_valid)

    model = ATTFeedforward(5000, 50)

    criterion = MultiViewLoss()
    criterion_t = BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3)

    trainer = DomainAdaptationTrainer(model, criterion, BCELoss(), optimizer, scheduler, max_epochs)
    trainer.fit(training_generator, validation_generator, tgt_domain_data_set, dictionary)
