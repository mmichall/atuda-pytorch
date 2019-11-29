from typing import List
import numpy as np
from torch.utils.data import dataset
import pandas as pd

from helper.data import doc2one_hot
from helper.reader import AmazonDomainDataReader


class AmazonDomainDataSet(dataset.Dataset):

    def __init__(self, domain: str=None, is_labeled=False):
        self.domain = domain
        self.data: pd.DataFrame = AmazonDomainDataReader.read(domain, is_labeled)
        self.dict = {}
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = self.data.loc[index]
        return index, doc2one_hot(item.acl_processed, self.dict), item.sentiment, item.src

    def get(self, index):
        return self.data.iloc[index]

    def get_labeled_indxs(self):
        return [getattr(tuple, 'Index') for tuple in self.data.itertuples() if tuple.is_labeled]

    def append(self, item):
        item.name = self.length
        self.data = self.data.append(item)
        self.length = len(self.data)

    def summary(self, name):
        print("\n___________________________________________________")
        print("> \t {} data set summary \t ".format(name))
        print("> \t Eexamples count: {} \t ".format(len(self.data)))
        print("> \t Dict length: {} \t ".format(len(self.dict)))
        print("___________________________________________________")


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
