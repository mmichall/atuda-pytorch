import config
from data_set import SentimentDataSet, MultiDomainSentimentDataSet
from pprint import pprint

dataset = MultiDomainSentimentDataSet(config.SOURCE_DOMAIN, config.TARGET_DOMAIN, n=5000)

for item in dataset.src_ds:
    pprint(dataset.one_hot_sentence([word[0] for word in item.acl_processed]))