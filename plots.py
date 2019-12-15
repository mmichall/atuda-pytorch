from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_set import load_data, merge, AmazonDomainDataSet
import random

tsne = TSNE(n_components=2, random_state=0, perplexity = 50,n_iter=5000)

src_domain_data_set, tgt_domain_data_set = load_data("books", "kitchen")

batches = []
y = []
# Init training data

xx: AmazonDomainDataSet = merge([src_domain_data_set, tgt_domain_data_set])
xx.dict = src_domain_data_set.dict

for idx, batch_one_hot, labels, src in DataLoader(xx, batch_size=1, shuffle=True):
    for _i, batch in enumerate(batch_one_hot):
        if 1 not in batch.numpy():
            continue
        batches.append(batch.numpy())
        y.append(labels[_i][0].numpy())

c = list(zip(batches, y))
random.shuffle(c)
batches, y = zip(*c)

batches = batches[:2000]
y = y[:2000]

tsne_obj = tsne.fit_transform(batches)

plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=y)
plt.show()