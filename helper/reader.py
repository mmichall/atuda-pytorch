import pandas as pd
from os import path
import config


class AmazonDomainDataReader:
    labeled_data_fls = ["negative.review", "positive.review"]
    unlabeled_data_fls = ["unlabeled.review"]

    @staticmethod
    def read(domain: str, is_labeled) -> pd.DataFrame:
        data = {'acl_processed': [], 'sentiment': []}

        _path = path.join(config.DATA_PATH, domain)
        data_fls = AmazonDomainDataReader.labeled_data_fls if is_labeled else AmazonDomainDataReader.unlabeled_data_fls
        for file in data_fls:
            with open(path.join(_path, file), 'r') as f:
                content = f.readlines()

            for line in content:
                line = line.split()
                x = [tuple(x.split(':')) for x in line[:-1]]
                y = line[-1].split(':')[1]
                data['acl_processed'].append(x)
                data['sentiment'].append(1 if 'positive' == y else 0)

        return pd.DataFrame(data)
