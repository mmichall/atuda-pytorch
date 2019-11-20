import config
import os.path as path
from typing import List
from torch.utils.data import dataset
from pandas import get_dummies
import pandas as pd
from pprint import pprint


class AmazonDomainDataset(dataset.Dataset):

    def __init__(self, domain: str):
        self._dictionary = {}
        self._domain = domain
        self._data = self.__load_data__(domain)
        self._len = len(self._data)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._data[index]

    def __load_data__(self, domain: str) -> List:
        filename = path.join(config.DATA_PATH, domain, 'negative.review')
        data = []
        with open(filename, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.split()
            x = [x.split(':')[0] for x in line[:-1]]
            y = line[-1].split(':')[1]
            for word in x:
                if word not in self._dictionary and len(self._dictionary) < 5000:
                    self._dictionary[word] = len(self._dictionary)
            data.append((x, y))
        return data

    def get_one_hot(self, word):
        return self._dictionary.setdefault(word, '<OOV>')


