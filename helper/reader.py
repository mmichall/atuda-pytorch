import pandas as pd
from os import path
import config


class AmazonDomainDataReader:
    data_fls = ["negative.review", "positive.review", "unlabeled.review"]

    @staticmethod
    def read(domain: str) -> pd.DataFrame:
        data = {'review': [], 'acl_processed': [], 'sentiment': [], 'is_labeled': []}

        _path = path.join(config.DATA_PATH, domain)
        for file in AmazonDomainDataReader.data_fls:
            with open(path.join(_path, file), 'r') as f:
                content = f.readlines()

            for line in content:
                line = line.split()
                x = [tuple(x.split(':')) for x in line[:-1]]
                y = line[-1].split(':')[1]
                data['review'].append('<empty>')
                data['acl_processed'].append(x)
                data['sentiment'].append(1 if 'positive' == y else 0)
                data['is_labeled'].append(file != "unlabeled.review")

        return pd.DataFrame(data)
