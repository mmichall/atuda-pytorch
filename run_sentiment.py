import config
from pprint import pprint
from helper.data_loader import AmazonDomainDataset


dataset: AmazonDomainDataset = AmazonDomainDataset(config.SOURCE_DOMAIN)