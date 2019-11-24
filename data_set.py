from typing import List
import numpy as np
from torch.utils.data import dataset
import pandas as pd

from helper.data import doc2one_hot
from helper.reader import AmazonDomainDataReader


class AmazonDomainDataSet(dataset.Dataset):

    def __init__(self, domain: str, is_labeled):
        self.domain = domain
        self.data: pd.DataFrame = AmazonDomainDataReader.read(domain, is_labeled)
        self.dict = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.loc[index]
        return index, doc2one_hot(item.acl_processed, self.dict), item.sentiment

    def get(self, index):
        return self.data.iloc[index]

    def get_labeled_indxs(self):
        return [getattr(tuple, 'Index') for tuple in self.data.itertuples() if tuple.is_labeled]

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

    def __getitem__(self, index):
        index = self.ids[index]
        return self._data_set.__getitem__(index)

    def __len__(self):
        return len(self.ids)

    def append(self, item):
        _len = len(self._data_set.data)
        item.name = _len
        self.ids.append(_len)
        self._data_set.data = self._data_set.data.append(item)
