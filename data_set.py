import copy
import random
from typing import List
import pandas as pd
from torch.utils.data import dataset, DataLoader
from utils.data import doc2one_hot, build_dictionary, train_valid_split
from utils.reader import AmazonDomainDataReader


class AmazonDomainDataSet(dataset.Dataset):

    def __init__(self, domain: str = None, is_labeled=False, denoising_factor=0., words_to_reconstruct=None):
        self.domain = domain
        self.data: pd.DataFrame = AmazonDomainDataReader.read(domain, is_labeled)
        self.dict = {}
        self.words_to_reconstruct = words_to_reconstruct
        self.denoising_factor = denoising_factor
        self.length = len(self.data)
        self.is_labeled = is_labeled

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = self.data.loc[index]
        _doc2one_hot = doc2one_hot(item.acl_processed, self.dict)

        # TODO: Such a dirty code! Has to be refactored immediately!
        if self.denoising_factor != 0.0:
            _denoised = copy.deepcopy(_doc2one_hot)
            for i in range(len(_denoised)):
                if random.random() < self.denoising_factor:
                    _denoised[i] = 0
            return index, _denoised, _doc2one_hot, item.src

        if self.words_to_reconstruct is not None:
            _denoised = doc2one_hot(item.acl_processed, self.dict, self.words_to_reconstruct)
            return index, _denoised, item.sentiment, self.is_labeled

        return index, _doc2one_hot, item.sentiment, item.src

    def get(self, index):
        return self.data.iloc[index]

    def get_labeled_indxs(self):
        return [getattr(tuple, 'Index') for tuple in self.data.itertuples() if tuple.is_labeled]

    def append(self, item):
        item.name = self.length
        self.data = self.data.append(item)
        self.length = len(self.data)

    def append_set(self, dataset):
        self.data = self.data.append(dataset.data, ignore_index=True)
        self.length = len(self.data)

    def summary(self, name):
        print("> {} data set summary \t ".format(name))
        print("+\t Examples count: {} \t ".format(len(self.data)))
        print("+\t Dict length: {} \t\n".format(len(self.dict)))


class AmazonSubsetWrapper(dataset.Dataset):

    def __init__(self, amazon_data_set: AmazonDomainDataSet, ids: List):
        self._data_set = amazon_data_set
        self.ids = ids
        self.length = len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]
        return self._data_set.__getitem__(index)

    def __len__(self):
        return self.length

    def append(self, item):
        _len = len(self._data_set.data)
        item.name = _len
        self.ids.append(_len)
        self._data_set.append(item)


def train_valid_target_split(src_domain_data_set, tgt_domain_data_set, params_train, train_valid_ratio=0.1) -> (DataLoader, DataLoader, DataLoader):
    #print(src_domain_data_set.data)

    train_idxs, valid_idxs = train_valid_split(0, len(src_domain_data_set), train_valid_ratio)
    print("Training set length: {}, Validation set length: {}".format(len(train_idxs), len(valid_idxs)))

    train_subset = AmazonSubsetWrapper(src_domain_data_set, train_idxs)
    valid_subset = AmazonSubsetWrapper(src_domain_data_set, valid_idxs)

    params_valid = {'batch_size': len(valid_subset)}
    params_target = {'batch_size': len(tgt_domain_data_set)}

    # Generators
    training_generator = DataLoader(train_subset, **params_train)
    validation_generator = DataLoader(valid_subset, **params_train)
    target_generator = DataLoader(tgt_domain_data_set, **params_target)

    return training_generator, validation_generator, target_generator


def as_one_dataloader(src_domain: str, tgt_domain: str, params_train, denoising_factor=0.0,
                      words_to_reconstruct=None) -> DataLoader:
    src_domain_data_set, tgt_domain_data_set = load_data(src_domain, tgt_domain)

    data_set = merge([src_domain_data_set, tgt_domain_data_set])
    data_set.dict = src_domain_data_set.dict
    data_set.denoising_factor = denoising_factor
    data_set.words_to_reconstruct = words_to_reconstruct
    data_set.summary('Unsupervised data set')

    return DataLoader(data_set, **params_train)


def load_data(src_domain, tgt_domain, verbose=False):
    src_domain_data_set = AmazonDomainDataSet(src_domain, True)
    tgt_domain_data_set = AmazonDomainDataSet(tgt_domain, False)

    dictionary = build_dictionary([src_domain_data_set, tgt_domain_data_set], 5000)

    src_domain_data_set.dict = dictionary
    tgt_domain_data_set.dict = dictionary

    if verbose:
        src_domain_data_set.summary('src_domain_data_set')
        tgt_domain_data_set.summary('tgt_domain_data_set')

    return src_domain_data_set, tgt_domain_data_set


def merge(datasets: List[AmazonDomainDataSet]) -> AmazonDomainDataSet:
    merged = AmazonDomainDataSet()
    for dataset in datasets:
        merged.append_set(dataset)

    return merged
