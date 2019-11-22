from typing import List, Dict
import pandas as pd
import numpy as np

from tqdm import tqdm
import random


def filter(list: List[str], dict: Dict) -> List[str]:
    return [item for item in list if item in dict]


def train_valid_split(range_from: int, range_to: int, valid_split: float):
    count = int((range_to - range_from) * valid_split)
    idxs = range(range_from, range_to)
    valid_idxs = random.sample(idxs, count)
    train_idxs = [idx for idx  in idxs if idx not in valid_idxs]
    return train_idxs, valid_idxs


def word2one_hot(word: str, dictionary: Dict, vec_length: int):
    one_hot_vec = np.zeros(vec_length)
    if word in dictionary:
        one_hot_vec[dictionary[word][0]] = 1
    return one_hot_vec


def doc2one_hot(doc: List[str], dictionary: Dict):
    vec_length = len(dictionary)
    vector = np.zeros(len(dictionary))
    for word in doc:
        vector = np.add(vector, word2one_hot(word[0], dictionary, vec_length))
    return vector


def build_dictionary(data_sets: List, limit=None):
    _dict = {}
    print("# Dictionary building... ")
    for item in tqdm(pd.concat([_set.data for _set in data_sets]).itertuples()):
        for word in item.acl_processed:
            cnt = _dict.setdefault(word[0], 0)
            _dict[word[0]] = cnt + int(word[1])

    _sorted_dict = _dict
    if limit:
        _sorted_dict = sorted(_dict.items(), key=lambda kv: kv[1])
        _sorted_dict.reverse()
        _sorted_dict = dict(_sorted_dict[:limit])

    for i, key in enumerate(_sorted_dict):
        _sorted_dict[key] = (i, _sorted_dict[key])
    return _sorted_dict
