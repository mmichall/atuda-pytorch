from torch.utils.data import dataset
import pandas as pd
from helper.reader import MultiDomainSentimentDataReader
from helper.data import filter
from tqdm import tqdm
from typing import List
from pprint import pprint
import numpy as np


class SentimentDataSet(dataset.Dataset):

    def __init__(self, domain: str, n=None):
        self.domain = domain
        self.n = n
        self.data: pd.DataFrame = MultiDomainSentimentDataReader.read(domain)
        self._dict = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        return self.one_hot_sentence([item[0] for item in item.acl_processed]), 1 if 'positive' == item.sentiment else 0

    def one_hot_word(self, word: str):
        if word in self._dict:
            vector = np.zeros(self.n)
            vector[self._dict[word][0]] = 1
            return vector
        else:
            return np.zeros(self.n)

    def one_hot_sentence(self, words: List[str]):
        vector = np.zeros(self.n)
        for word in words:
            vector = np.add(vector, self.one_hot_word(word))
        return vector


class MultiDomainSentimentDataSet:

    def __init__(self, src_domain_nm: str, tgt_domain_nm: str, n=None):
        print("# MultiDomainSentimentDataSet building [source domain: {}, target domain: {}]".format(src_domain_nm, tgt_domain_nm))
        self.src_ds = SentimentDataSet(src_domain_nm, n)
        self.tgt_ds = SentimentDataSet(tgt_domain_nm, n)
        self.n = n
        self._dict = self._build_dictionary(n)
        self.src_ds._dict = self._dict
        self.tgt_ds._dict = self._dict

    def _build_dictionary(self, n=None):
        _dict = {}
        print("# Dictionary building... ")
        for item in tqdm(pd.concat([self.src_ds.data, self.tgt_ds.data]).itertuples()):
            for word in item.acl_processed:
                cnt = _dict.setdefault(word[0], 0)
                _dict[word[0]] = cnt + int(word[1])

        _sorted_dict = _dict
        if n:
            _sorted_dict = sorted(_dict.items(), key=lambda kv: kv[1])
            _sorted_dict.reverse()
            _sorted_dict = dict(_sorted_dict[:n])

        for i, key in enumerate(_sorted_dict):
            _sorted_dict[key] = (i, _sorted_dict[key])
        return _sorted_dict







