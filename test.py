# Simply test (SVM) for all thresholds and all cases (12)
# An impact of threshold (a comparison of kl plot)

# AE hidden layer as inition state/ embedding for perceptron (another simply neural model)
# Domain Loss
import ast
from enum import Enum
import glob

import torch
from pandas import np
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from data_set import load_data, train_valid_target_split
from nn.model import SimpleAutoEncoder

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
        x, y = self._to_array(self.train_data_loader)
        x_tgt, y_tgt = self._to_array(self.target_data_loader)
        for _ in range(times):
            self.fit(x, y)
            acc = self.measure(x_tgt, y_tgt)
            measures.append(acc)
        print(['> Test [{} > {} {}]'.format(self.src_domain, self.tgt_domain, th), measures, np.mean(measures)])

    def run(self, times=1, pretrained_path='tmp/embedd'):
        arr = glob.glob("tmp/embedd/auto_encoder_{}_{}*.pt".format(self.src_domain, self.tgt_domain))
        arr.sort(key=lambda x: x.split('_')[10])
        arr = ['_'] + arr
        for embedd_path in arr:
            th = 'NONE'
            if embedd_path != '_':
                th = embedd_path.split('_')[10]
                self.with_embedding(embedd_path)
                print(embedd_path)
            self._run(th, times)


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

    def fit(self, X, Y):
        self.cls = LinearSVC(max_iter=4000)
        self.cls.fit(X, Y)

    def measure(self, X_tgt, Y_tgt):
        ground_truth = Y_tgt
        predicted = self.cls.predict(X_tgt)
        return np.round((ground_truth == predicted).sum() / len(ground_truth), 3)


def get_ae_model(path: str):
    ae_embedding_path = None
    if path is not None:
        ae_embedding_path = path
        ae_model = SimpleAutoEncoder(ast.literal_eval('(5000, 3000)'))
        ae_model.load_state_dict(torch.load(path))

    if not ae_embedding_path:
        raise RuntimeError('Failed to load AutoEncoder embedding')
    ae_model.froze()
    return ae_model


if __name__ == '__main__':
    results = []
    results.append(Test('books', 'dvd').run(3))
    results.append(Test('books', 'kitchen').run(3))
    results.append(Test('books', 'electronics').run(3))

    results.append(Test('kitchen', 'dvd').run(3))
    results.append(Test('kitchen', 'electronics').run(3))
    results.append(Test('kitchen', 'books').run(3))

    results.append(Test('dvd', 'kitchen').run(3))
    results.append(Test('dvd', 'electronics').run(3))
    results.append(Test('dvd', 'books').run(3))

    results.append(Test('electronics', 'kitchen').run(3))
    results.append(Test('electronics', 'dvd').run(3))
    results.append(Test('electronics', 'books').run(3))


    for result in results:
        print(result)





