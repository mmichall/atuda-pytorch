from typing import List

from data_set import AmazonDomainDataSet


def merge(datasets: List[AmazonDomainDataSet]) -> AmazonDomainDataSet:
    merged = AmazonDomainDataSet()
    for dataset in datasets:
        merged.append_set(dataset)

    return merged