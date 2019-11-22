from typing import List

from torch.utils.data import dataset
import pandas as pd
from helper.reader import AmazonDomainDataReader
from helper.data import doc2one_hot


class AmazonDomainDataSet(dataset.Dataset):

    def __init__(self, domain: str):
        self.domain = domain
        self.data: pd.DataFrame = AmazonDomainDataReader.read(domain)
        self.dict = {}
        self.labeled_indxs = self.get_labeled_indxs()

    def __len__(self):
        return len(self.labeled_indxs)

    def __getitem__(self, index):
        index = self.labeled_indxs[index]
        item = self.data.iloc[index]
        return doc2one_hot(item.acl_processed, self.dict), item.sentiment

    def get_labeled_indxs(self):
        return [getattr(tuple, 'Index') for tuple in self.data.itertuples() if tuple.is_labeled]

    def summary(self, name):
        print("\n___________________________")
        print("> \t {} data set summary \t ".format(name))
        print("> \t Labeled examples count: {} \t ".format(len(self.labeled_indxs)))
        print("> \t Unlabeled examples count: {} \t ".format(len(self.data) - len(self.labeled_indxs)))
        print("> \t Dict length: {} \t ".format(len(self.dict)))
        print("___________________________")


class AmazonDomainSubset(dataset.Dataset):

    def __init__(self, amazon_data_set: AmazonDomainDataSet, ids: List):
        self._data_set = amazon_data_set
        self.ids = ids

    def __getitem__(self, index):
        index = self.ids[index]
        return self._data_set.__getitem__(index)

    def __len__(self):
        return len(self.ids)
