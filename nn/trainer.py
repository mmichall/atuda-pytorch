import sys
import copy
from statistics import mean
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import math
import torch.nn.functional as F
from data_set import AmazonDomainDataSet
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
                n_epochs_stop = 0
                prev_loss = _loss_mean

            if n_epochs_stop == self.epochs_no_improve:
                print('Early Stopping!')
                break

            print('')
        print("> Training is over. Thank you for your patience :).")


class DomainAdaptationTrainer:

    def __init__(self, model, criterion, criterion_t, optimizer, scheduler, max_epochs, ae_model=None, epochs_no_improve=3):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.ae_model = ae_model
        self.epochs_no_improve = epochs_no_improve
        self.model.to(device)

    def __fit(self, training_generator, validation_generator, target_generator):
        self._fit(training_generator, validation_generator, target_generator, self.max_epochs)
        torch.save(self.model.state_dict(), 'tmp/model_5000+250.pt')
        # self.model.load_state_dict(torch.load('tmp/model_5000_100.pt'))

        params = {'batch_size': 8,
                  'shuffle': False,
                  'num_workers': 1}

        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = target_generator.dataset.dict

        # Pseudo labeling
        _tgt_len = len(target_generator)

        idxs_to_remove = []
        wrong_target = 0
        for _i in range(1, 2):
            ######################################
            # training_tgt_data_set = AmazonDomainDataSet()
            # training_tgt_data_set.dict = _dict
            ######################################
            batch_num = 0
            _len = int((_i / 1) * _tgt_len)
            target_data_set.length = _tgt_len
            target_generator = DataLoader(target_data_set, **{'batch_size': _tgt_len, 'shuffle': True})
            with torch.set_grad_enabled(False):
                self.model.eval()
                for idx, batch_one_hot, labels in target_generator:
                    # Transfer to GPU
                    labels = torch.stack(labels, dim=1)
                    batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

                    if self.ae_model:
                        batch_one_hot = self.ae_model(batch_one_hot)
                    batch_num += 1

                    f1, f2, ft = self.model(batch_one_hot)
                    t = Variable(torch.FloatTensor([0.5]))  # threshold
                    out1 = (f1 > t)
                    out2 = (f2 > t)
                    outt = (ft > t)

                    _f1 = f1.cpu().numpy().flatten()
                    _f2 = f2.cpu().numpy().flatten()
                    _idx = idx.cpu().numpy().flatten()

                    for i in range(0, len(outt)):
                        if _idx[i] in idxs_to_remove:
                            continue
                        #  if np.array_equal(out1[i].cpu().numpy(), out2[i].cpu().numpy())\
                        #         and max(f1[i]) > 0.95 \
                        #         and max(f2[i]) > 0.95:
                        if True:
                            item = copy.deepcopy(target_generator.dataset.get(_idx[i]))
                            x1 = outt[i].cpu().numpy()
                            x2 = np.asarray(item.sentiment)
                            if not np.array_equal(x1, x2):
                                wrong_target += 1
                            item.sentiment = outt[i].cpu().numpy()
                            training_tgt_data_set.append(item)
                            idxs_to_remove.append(_idx[i])
                            if len(training_tgt_data_set) == _len:
                                break

            training_tgt_data_loader = DataLoader(training_tgt_data_set, **params)
            print('Wrong labeled: {} on {}'.format(wrong_target, len(idxs_to_remove)))

            # Step 2
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=3)
            self._fit(training_tgt_data_loader, target_generator, max_epochs=20, is_step2=True)

    def fit(self, training_loader: DataLoader, validation_loader: DataLoader, target_generator: DataLoader = None,
            max_epochs=10, is_step2=False, _dict=None):
        n_epochs_stop = 0
        prev_metric = 0

        self.model.train()
        for epoch in range(max_epochs):
            print('+ \tepoch number: ' + str(epoch))
            n_batch = 0
            loss_f1f2 = []
            loss_t = []
            acc_all = []
            batches = []
            batches_n = math.ceil(len(training_loader.dataset) / training_loader.batch_size)
            # Init training data
            for idx, batch_one_hot, labels, src in training_loader:
                batches.append((idx, batch_one_hot, labels, 1))

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
                # ae_output = ae_output.to(device, torch.float)

                self.optimizer.zero_grad()

                f1, f2, ft = self.model(input)

                _loss_f1f2 = self.criterion(f1, f2, self.model.f1_1.weight, self.model.f2_1.weight, labels_)
                _loss_t = self.criterion_t(ft, labels_)
                loss_f1f2.append(_loss_f1f2.item())
                loss_t.append(_loss_t.item())

                _acc = acc(F.softmax(ft, dim=1), labels)
                acc_all.append(_acc)
                _loss_f1f2_mean = round(mean(loss_f1f2), 4)
                _loss_t_mean = round(mean(loss_f1f2), 4)

                sys.stdout.write(
                    '\r+\tbatch: {} / {}, {}: {}, acc: {}'.format(n_batch, batches_n, self.criterion.__class__.__name__,
                                                         _loss_f1f2_mean, round(mean(acc_all), 4)))

                _loss_f1f2.backward(retain_graph=True)
                self.optimizer.step()
                if not is_step2 or not src:
                    _loss_t.backward()
                    self.optimizer.step()

            sys.stdout.write('\n')

            _valid_acc = self.valid(validation_loader, target_generator)
            self.scheduler.step(_loss_f1f2_mean, epoch)

            if _valid_acc <= prev_metric:
                n_epochs_stop += 1
                if n_epochs_stop == self.epochs_no_improve:
                    print('Early stopping!')
                    break
            else:
                n_epochs_stop = 0
                prev_metric = _valid_acc


    def valid(self, data_loader, data_loader_2=None):
        with torch.set_grad_enabled(False):
            self.model.eval()
            _acc_all = []
            _acc_all_tgt = []
            for idx, local_batch, local_labels, src in data_loader:
                local_labels = torch.stack(local_labels, dim=1)
                local_batch = local_batch.to(device, torch.float)
                ae_output = self.ae_model(local_batch)
                ae_output = ae_output.to(device)
                f1, f2, ft = self.model(torch.cat([local_batch, ae_output], 1))
                _acc_all.append(acc(F.softmax(ft, dim=1), local_labels))
            if data_loader_2 is not None:
                for idx, local_batch, local_labels, src in data_loader_2:
                    local_labels = torch.stack(local_labels, dim=1)
                    local_batch = local_batch.to(device, torch.float)
                    ae_output = self.ae_model(local_batch)
                    ae_output = ae_output.to(device)
                    f1, f2, ft = self.model(torch.cat([local_batch, ae_output], 1))
                    _acc_all_tgt.append(acc(F.softmax(ft, dim=1), local_labels))
                print('\r+\tacc valid: {}, acc target: {}\n'.format(round(mean(_acc_all), 4),
                                                                         round(mean(_acc_all_tgt), 4)))
            else:
                print('\r+\tacc valid: {}'.format(round(mean(_acc_all), 4)))

            return mean(_acc_all)
