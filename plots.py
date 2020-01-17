import ast

import pickle
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_set import load_data, merge, AmazonDomainDataSet
import random

from nn.model import SimpleAutoEncoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

ae_model = SimpleAutoEncoder(ast.literal_eval('(5000, 500, 250)'))
ae_model.load_state_dict(torch.load('tmp/auto_encoder_5000_500_250_bce.pt'))
ae_model.set_train_mode(False)
ae_model.to(device)

tsne = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=5000)

src_domain_data_set, tgt_domain_data_set = load_data("books", "kitchen")

batches = []
y = []
# Init training data

xx: AmazonDomainDataSet = merge([src_domain_data_set, tgt_domain_data_set])
xx.dict = src_domain_data_set.dict

for idx, batch_one_hot, labels, src in DataLoader(xx, batch_size=1, shuffle=False):
    for _i, batch in enumerate(batch_one_hot):
        if 1 not in batch.cpu().numpy():
            continue
        batches.append(ae_model(batch.to(device).float()).detach().cpu().numpy())
        # y.append(labels[_i][0].detach().cpu().numpy())
        y.append(src[_i].detach().cpu().numpy())

c = list(zip(batches, y))
random.shuffle(c)
batches, y = zip(*c)

batches = batches[:5000]
y = y[:5000]

tsne_obj = tsne.fit_transform(batches)

print(tsne_obj)

with open('tmp/tsne.pck', 'wb') as f, open('tmp/tsne_y.pck', 'wb') as fy:
    pickle.dump(tsne_obj, f)
    pickle.dump(y, fy)

plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=y)
plt.show()