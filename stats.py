from data_set import AmazonDomainDataSet, train_valid_target_split, load_data
from utils.data import build_dictionary
import numpy as np

def get_unique_per_set_words(data_set1: AmazonDomainDataSet, data_set2: AmazonDomainDataSet):

    dictionary = build_dictionary([data_set1, data_set2], 5000)

    set1_words = set()
    set2_words = set()
    for i, row in data_set1.data.iterrows():
        for word in row.acl_processed:
            if word[0] in dictionary:
                set1_words.add(word[0])

    for i, row in data_set2.data.iterrows():
        for word in row.acl_processed:
            if word[0] in dictionary:
                set2_words.add(word[0])

    return set([word for word in set1_words if word not in set2_words]).union(set([word for word in set2_words if word not in set1_words]))



src_domain_data_set, tgt_domain_data_set = load_data('books', 'kitchen', verbose=True)
get_unique_per_set_words(src_domain_data_set, tgt_domain_data_set)
