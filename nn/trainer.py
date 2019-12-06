import random
import sys
import copy
from statistics import mean
import operator
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import math
import torch.nn.functional as F
from data_set import AmazonDomainDataSet, train_valid_target_split
from nn.model import ModelWithTemperature
from utils.measure import acc
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class AutoEncoderTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, max_epochs, epochs_no_improve=3):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.epochs_no_improve = epochs_no_improve

    def fit(self, train_data_generator):
        print("> Training is running...")
        self.model.train()
        for epoch in range(self.max_epochs):
            _loss = 0
            _batch = 0
            prev_loss = 999
            prev_model = None
            batches_n = math.ceil(len(train_data_generator.dataset) / train_data_generator.batch_size)
            print('+ \tepoch number: ' + str(epoch))
            for idx, inputs, labels in train_data_generator:
                _batch += 1
                inputs, labels = inputs.to(device, torch.float), labels.to(device, torch.float)
                self.optimizer.zero_grad()

                out = self.model(inputs)
                loss = self.criterion(out, labels)

                _loss += loss.item()
                _loss_mean = round(_loss / _batch, 4)
                sys.stdout.write(
                    '\r+\tbatch: {} / {}, {}: {}'.format(_batch, batches_n, self.criterion.__class__.__name__,
                                                         _loss_mean))
                loss.backward()
                self.optimizer.step()
            if prev_loss <= _loss_mean:
                n_epochs_stop += 1
            else:
                prev_model = self.model
                n_epochs_stop = 0
                prev_loss = _loss_mean

            if n_epochs_stop == self.epochs_no_improve:
                # self.model = prev_model
                print('Early Stopping!')
                break

            print('')
        print("> Training is over. Thank you for your patience :).")


class DomainAdaptationTrainer:

    def __init__(self, model, criterion, criterion_t, optimizer, scheduler, max_epochs, ae_model=None,
                 epochs_no_improve=3):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.ae_model = ae_model
        self.epochs_no_improve = epochs_no_improve
        self.model.to(device)
        self.calibrated_model = self.model

    def fit(self, training_loader: DataLoader, validation_loader: DataLoader, target_data_set: DataLoader = None,
            max_epochs=10, is_step2=False, _dict=None):
        n_epochs_stop = 0
        prev_metric = 0

        for epoch in range(max_epochs):
            self.model.train()
            print('+ \tepoch number: ' + str(epoch))
            n_batch = 0
            loss_f1f2 = []
            acc_all = []
            batches = []
            # Init training data
            for idx, batch_one_hot, labels, src in training_loader:
                batches.append((idx, batch_one_hot, labels, True))

            if is_step2 and target_data_set is not None:
                target_generator = DataLoader(target_data_set, batch_size=training_loader.batch_size)
                for idx, batch_one_hot, labels, src in target_generator:
                    batches.append((idx, batch_one_hot, labels, False))

            random.shuffle(batches)

            for idx, input, labels, src in batches:
                n_batch += 1

                if type(labels) == list:
                    labels = torch.stack(labels, dim=1)

                # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices
                # max(1) will return the maximal value (and index in PyTorch) in this particular dimension.
                input, labels_ = input.to(device, torch.float), torch.max(labels, 1)[1].to(device, torch.long)
                if self.ae_model:
                    ae_output = self.ae_model(input)
                    input = torch.cat([input, ae_output], 1)

                self.optimizer.zero_grad()

                f1, f2, ft = self.model(input)
                _loss = self.criterion(f1, f2, self.model.get_f1_1().weight, self.model.get_f2_1().weight, labels_)

                if not is_step2 or not src:
                    _loss_t = self.criterion_t(ft, labels_)
                    _loss = _loss + _loss_t

                loss_f1f2.append(_loss.item())
               # loss_t.append(_loss_t.item())

                _acc = (acc(F.softmax(f1, dim=1), labels) + acc(F.softmax(f2, dim=1), labels)) / 2
                acc_all.append(_acc)
                _loss_f1f2_mean = round(mean(loss_f1f2), 4)
               # _loss_t_mean = round(mean(loss_f1f2), 4)

                sys.stdout.write(
                    '\r+\tbatch: {}, {}: {}, acc: {}'.format(n_batch, self.criterion.__class__.__name__,
                                                                  _loss_f1f2_mean, round(mean(acc_all), 4)))

                _loss.backward()
                self.optimizer.step()
            sys.stdout.write('\n')

            _valid_acc = self.valid(validation_loader)
            self.scheduler.step(_valid_acc)

            if _valid_acc <= prev_metric:
                n_epochs_stop += 1
                if n_epochs_stop == self.epochs_no_improve:
                    print('Early stopping!')
                    break
            else:
                n_epochs_stop = 0
                prev_metric = _valid_acc

        # calibrating
        self.calibrated_model = ModelWithTemperature(self.model, self.ae_model)
        self.calibrated_model.set_temperature(validation_loader)

    def valid(self, data_loader, data_loader_2=None):
        with torch.set_grad_enabled(False):
            self.model.eval()
            self.calibrated_model.eval()
            _acc_all = []
            _acc_all_tgt = []
            for idx, input, local_labels, src in data_loader:
                local_labels = torch.stack(local_labels, dim=1)
                input = input.to(device, torch.float)
                if self.ae_model:
                    ae_output = self.ae_model(input)
                    input = torch.cat([input, ae_output], 1)
                f1, f2, ft = self.calibrated_model(input)
                _acc_all.append(acc(F.softmax(ft, dim=1), local_labels))
            if data_loader_2 is not None:
                for idx, input, local_labels, src in data_loader_2:
                    local_labels = torch.stack(local_labels, dim=1)
                    input = input.to(device, torch.float)
                    if self.ae_model:
                        ae_output = self.ae_model(input)
                        input = torch.cat([input, ae_output], 1)
                    f1, f2, ft = self.calibrated_model(input)
                    _acc_all_tgt.append(acc(F.softmax(ft, dim=1), local_labels))
                print('\r+\tacc valid: {}, acc target: {}\n'.format(round(mean(_acc_all), 4),
                                                                    round(mean(_acc_all_tgt), 4)))
            else:
                print('\r+\tacc valid: {}'.format(round(mean(_acc_all), 4)))

            return mean(_acc_all)

    def pseudo_label(self, src_generator: DataLoader, target_data_set, train_params, iterations=20, max_epochs=3):
        target_generator = DataLoader(target_data_set, batch_size=len(target_data_set))
       # src_generator = DataLoader(src_data_set, **train_params)
        idx_added = []
        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = target_data_set.dict

        for _iter in range(iterations):
            n_wrong_labeled = 0
            n_all_qualified = 0
            LIMIT = int(len(target_generator.dataset) / iterations)

            idx, f1, f2, _ = self._predict(target_generator)
            f1_max, f1_idx = torch.max(f1, dim=1)
            f2_max, f2_idx = torch.max(f2, dim=1)

            idx = idx.numpy()
            f_sum = f1_max + f2_max
            f_sum = f_sum.cpu().numpy()
            idxs = {}
            for i in idx:
                idxs[i] = f_sum[i]
            idxs = sorted(idxs.items(), key=operator.itemgetter(1))
            idxs.reverse()

            most_confident_idxs = []

            for _i, _ in idxs:
                if _i not in idx_added:
                    most_confident_idxs.append(_i)
                    if len(most_confident_idxs) == LIMIT:
                        break

            for _i in most_confident_idxs:
                if f1_idx[_i] == f2_idx[_i]:# and f1_max[_i] > 0.90 and f2_max[_i] > 0.90:
                    n_all_qualified += 1
                    idx_added.append(_i)
                    item = copy.deepcopy(target_generator.dataset.get(_i))
                    if np.argmax(item.sentiment) != np.argmax(f1[_i].cpu().numpy()):
                        n_wrong_labeled += 1
                    item.sentiment = (1, 0) if f1_idx[_i] == 0 else (0, 1)
                    training_tgt_data_set.append(item)
                else:
                    pass
            print('Wrong labeled count: {} on {} qualified, all {}'.format(n_wrong_labeled, n_all_qualified, len(idx)))

           # train_generator, valid_generator, target_generator = train_valid_target_split(training_tgt_data_set,
             #                                                                             target_data_set,
               #                                                                           train_params)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=2)

           # src_generator.dataset.length = (iterations - (_iter + 1)) * int((src_generator.dataset.length / iterations))
            self.fit(src_generator, target_generator, training_tgt_data_set, is_step2=True, max_epochs=6)

    def _predict(self, data_generator: DataLoader):
        assert data_generator.batch_size == len(data_generator.dataset)
        self.model.eval()
        with torch.set_grad_enabled(False):
            for idx, input, labels, src in data_generator:
                input = input.to(device, torch.float)
                if self.ae_model:
                    ae_output = self.ae_model(input)
                    input = torch.cat([input, ae_output], 1)
                f1, f2, ft = self.calibrated_model(input)
                return idx, F.softmax(f1, dim=1), F.softmax(f2, dim=1), F.softmax(ft, dim=1)
