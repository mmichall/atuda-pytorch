# Simply test (SVM) for all thresholds and all cases (12)
# An impact of threshold (a comparison of kl plot)

# AE hidden layer as inition state/ embedding for perceptron (another simply neural model)
# Domain Loss
import ast
from collections import defaultdict
from enum import Enum
import glob
from random import shuffle

import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_set import load_data, AmazonSubsetWrapper
from nn.model import SimpleAutoEncoder
from nn.trainer import DistNetTrainer
from utils.data import train_valid_split

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class PretrainedEmbeddingType(Enum):
    NONE = 1
    AUTO_ENCODER = 2


class Test:

    def __init__(self, src_domain: str, tgt_domain: str):
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        src_domain_data_set, tgt_domain_data_set = load_data(src_domain, tgt_domain, verbose=False)
        self.train_data_loader = DataLoader(src_domain_data_set,
                                            **{'batch_size': len(src_domain_data_set), 'shuffle': True})
        self.target_data_loader = DataLoader(tgt_domain_data_set,
                                             **{'batch_size': len(tgt_domain_data_set), 'shuffle': True})
        self.embedding_path = None
        self.embedding_model = None

    def with_embedding(self, embedding_path):
        self.embedding_path = embedding_path
        self.embedding_model = get_ae_model(embedding_path)
        return self

    def _run(self, th: str, times=1):
        measures = []
        b_measures = []
        x, y = self._to_array(self.train_data_loader)
        x_tgt, y_tgt = self._to_array(self.target_data_loader)
        for _ in range(times):
            self.fit(x, y)
            _, _, b_acc = self.measure(x, y)
            non_zeros, zeros, acc = self.measure(x_tgt, y_tgt)
            self.pca(x, x_tgt, y, y_tgt)
            measures.append(acc)
            b_measures.append(b_acc)
        print(['> Test [{} > {} {}. Acc: all {}; avg {}; src acc {}]'.format(self.src_domain, self.tgt_domain, th, measures, np.mean(measures), np.mean(b_measures))])
        return np.mean(measures)

    def pca(self, x, x2, y, y2):
        pca = LinearDiscriminantAnalysis()
        X = np.concatenate((x, x2), axis=0)
        Y = np.concatenate((y, y2), axis=0)
        #
        # temp = list(zip(X, Y))
        # shuffle(temp)
        # X, Y = zip(*temp)
        pca_result = pca.fit_transform(X, Y)

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        plt.figure(figsize=(16, 10))
        df = pd.DataFrame()

        # df['pca-one'] = pca_result[:, 0]
        # df['pca-two'] = pca_result[:, 1]
        # df['y'] = y2
        #
        # import seaborn as sns
        # sns.scatterplot(
        #     x="pca-one", y="pca-two",
        #     palette=sns.color_palette("hls", 2),
        #     data=df,
        #     hue="y",
        #     legend="full",
        #     alpha=0.3
        # )
        #
        # plt.show()


    def run(self, times=1, pretrained_path='tmp/embedd'):
            arr = glob.glob("tmp/embedd/auto_encoder_{}_{}*.pt".format(self.src_domain, self.tgt_domain))
            arr.sort(key=lambda x: x.split('_')[10])
            arr = ['_'] + arr
            accs = []
            th = 'NONE'

            mapped = defaultdict(lambda: [])
            for embedd_path in arr:
                if embedd_path != '_':
                    th = embedd_path.split('_')[9]
                mapped[th].append(embedd_path)
            for th, embedd_paths in mapped.items():
                for embedd_path in embedd_paths:
                    if th != 'NONE':
                        self.with_embedding(embedd_path)
                    accs.append(self._run(th, times))
                print('|-------------------- SUMMARY --------------------|')
                print(np.round(np.mean(accs), 3), '\n')
                accs = []


    def _to_array(self, data_generator):
        X = []
        Y = []
        for idx, batch_one_hot, labels, src in data_generator:
            y = labels[0].numpy()
            if self.embedding_model:
                batch_one_hot = batch_one_hot.to(device, torch.float)
                batch_one_hot = self.embedding_model(batch_one_hot).cpu().detach().numpy()
            x = batch_one_hot
            X.append(x)
            Y.append(y)

        return np.concatenate(X), np.concatenate(Y)

    def fit(self, X=None, Y=None):
        self.cls = LinearSVC(max_iter=20000, dual=False)
        self.cls.fit(X, Y)
        # self.cls = LogisticRegression(max_iter=2000)
        # src_domain_data_set, tgt_domain_data_set = load_data("dvd", "electronics", verbose=True, return_input=False)
        #
        # self.trainer = DistNetTrainer("dvd", "electronics", max_epochs=1000)
        #
        # self.trainer.fit(src_domain_data_set, tgt_domain_data_set, batch_size=32)
        # self.trainer.eval(tgt_domain_data_set)



    def measure(self, X_tgt, Y_tgt):
        ground_truth = Y_tgt
        predicted = self.cls.predict(X_tgt)
        return np.count_nonzero(ground_truth), ground_truth.size - np.count_nonzero(ground_truth), np.round((ground_truth == predicted).sum() / len(ground_truth), 3)


def get_ae_model(path: str):
    ae_embedding_path = None
    if path is not None:
        ae_embedding_path = path
        splitted = ae_embedding_path.split('_')
        ae_model = SimpleAutoEncoder(ast.literal_eval('({}, {})'.format(splitted[5], splitted[6])))
        ae_model.encoder.load_state_dict(torch.load(path))

    if not ae_embedding_path:
        raise RuntimeError('Failed to load AutoEncoder embedding')
    ae_model.froze()
    return ae_model


if __name__ == '__main__':
    results = []
    results.append(Test('books', 'dvd').run(1))
    results.append(Test('books', 'kitchen').run(1))
    results.append(Test('books', 'electronics').run(1))

    results.append(Test('kitchen', 'dvd').run(1))
    results.append(Test('kitchen', 'electronics').run(1))
    results.append(Test('kitchen', 'books').run(1))
    #
    results.append(Test('dvd', 'kitchen').run(1))
    results.append(Test('dvd', 'electronics').run(1))
    results.append(Test('dvd', 'books').run(1))
    #
    results.append(Test('electronics', 'kitchen').run(1))
    results.append(Test('electronics', 'dvd').run(1))
    results.append(Test('electronics', 'books').run(1))






