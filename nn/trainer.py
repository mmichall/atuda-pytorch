import sys
import copy
from statistics import mean
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from data_set import AmazonDomainDataSet
from helper.measure import acc
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
        self.epochs_no_improve = 3

    def fit(self, train_data_generator):
        self.model.train()
        for epoch in range(self.max_epochs):
            _loss = 0
            _batch = 0
            prev_loss = 999
            print('Epoch: ' + str(epoch))
            for idx, inputs, labels in train_data_generator:
                _batch += 1
                inputs, labels = inputs.to(device, torch.float), labels.to(device, torch.float)
                self.optimizer.zero_grad()

                out = self.model(inputs)
                loss = self.criterion(out, labels)

                _loss += loss.item()
                _loss_mean = round(_loss / _batch, 4)
                sys.stdout.write('\r MSE Error: ' + str(_loss_mean))
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


class DomainAdaptationTrainer:

    def __init__(self, ae_model, model, criterion, criterion_t, optimizer, scheduler, max_epochs):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.ae_model = ae_model
        self.model.to(device)

    def fit(self, training_generator, validation_generator, target_data_set, _dict):
        target_generator = DataLoader(target_data_set, **{'batch_size': len(target_data_set)})
        self._fit(training_generator, validation_generator, target_generator, self.max_epochs, _dict=_dict)
        torch.save(self.model.state_dict(), 'tmp/model_5000+250.pt')
        #self.model.load_state_dict(torch.load('tmp/model_5000_100.pt'))

        params = {'batch_size': 8,
                  'shuffle': False,
                  'num_workers': 1}

        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = _dict

        # Pseudo labeling
        _tgt_len = len(target_data_set)

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

    def _fit(self, data_loader: DataLoader, validation_loader: DataLoader, target_generator: DataLoader = None,
             max_epochs=10, src_data_loader: DataLoader = None, is_step2=False, _dict=None):
        self.model.train()
        # Loop over epochs
        epochs_no_improve = 3
        n_epochs_stop = 0
        _prev_metric = 0

        for epoch in range(max_epochs):
            # Training
            batch_num = 0
            loss_all = []
            _acc_all = []
            _acc_all1 = []
            _acc_all2 = []

            batches = []
            # if src_data_loader:
            #     for idx, batch_one_hot, labels in src_data_loader:
            #         batches.append((idx, batch_one_hot, labels, 1))

            for idx, batch_one_hot, labels in data_loader:
                batches.append((idx, batch_one_hot, labels, 0))

            random.shuffle(batches)
            for idx, batch_one_hot, labels, src in batches:
                batch_num += 1
                # Transfer to GPU
                if type(labels) == list:
                    labels = torch.stack(labels, dim=1)
                batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)
                ae_output = self.ae_model(batch_one_hot.cpu())
                ae_output = ae_output.to(device, torch.float)
                # Model computations
                self.optimizer.zero_grad()
                # Forward pass
                f1, f2, ft = self.model(torch.cat([batch_one_hot, ae_output], 1))
                # Compute Loss
                # f1_out, f2_out, w1, w2, y, _del
                loss = self.criterion(f1, f2, self.model.f1_1.weight, self.model.f2_1.weight, labels, 0.00001)
                loss_t = self.criterion_t(ft, labels)
                loss_all.append(loss.item())

                if is_step2:
                    _acc = acc(ft, labels)
                    _acc_all.append(_acc)
                    _loss_mean = round(mean(loss_all), 10)

                    sys.stdout.write(
                        '\r### Ft: Epoch {}, Batch {}, train loss: {} , acc: {} ###'.format(epoch, batch_num,
                                                                                            _loss_mean,
                                                                                            round(mean(_acc_all),
                                                                                                  4)))
                    sys.stdout.flush()
                else:
                    _loss_mean = round(mean(loss_all), 4)

                    _acc1 = acc(f1, labels)
                    _acc_all1.append(_acc1)

                    _acc2 = acc(f2, labels)
                    _acc_all2.append(_acc2)

                    sys.stdout.write(
                        '\r### Epoch {}, Batch {}, train loss: {} F1 acc: {} ### | ### F2 acc: {} ###'.format(
                            epoch, batch_num, _loss_mean, round(mean(_acc_all1), 4), round(mean(_acc_all2), 4)
                        ))

                    sys.stdout.flush()
                # Backward pass
                loss.backward(retain_graph=True)
                if not is_step2 or src == 0:
                    loss_t.backward()
                self.optimizer.step()
            sys.stdout.write('\n')

            _valid_acc = self.valid(validation_loader, target_generator)
            self.scheduler.step(_loss_mean, epoch)

            if _valid_acc <= _prev_metric:
                n_epochs_stop += 1
                if n_epochs_stop == epochs_no_improve:
                    print('Early stopping!')
                    break
            else:
                n_epochs_stop = 0
                _prev_metric = _valid_acc

    def valid(self, data_loader, data_loader_2=None):
        with torch.set_grad_enabled(False):
            self.model.eval()
            _acc_all = []
            _acc_all_tgt = []
            for idx, local_batch, local_labels in data_loader:
                local_labels = torch.stack(local_labels, dim=1)
                local_batch = local_batch.to(device, torch.float)
                ae_output = self.ae_model(local_batch.cpu())
                ae_output = ae_output.to(device)
                f1, f2, ft = self.model(torch.cat([local_batch, ae_output], 1))
                _acc_all.append(acc(ft, local_labels))
            if data_loader_2 is not None:
                for idx, local_batch, local_labels in data_loader_2:
                    local_labels = torch.stack(local_labels, dim=1)
                    local_batch = local_batch.to(device, torch.float)
                    ae_output = self.ae_model(local_batch.cpu())
                    ae_output = ae_output.to(device)
                    f1, f2, ft = self.model(torch.cat([local_batch, ae_output], 1))
                    _acc_all_tgt.append(acc(ft, local_labels))
                print('\r acc SOURCE_VALID: {}, acc TARGET: {}\n'.format(round(mean(_acc_all), 4), round(mean(_acc_all_tgt), 4)))
            else:
                print('\r acc SOURCE_VALID: {}'.format(round(mean(_acc_all), 4)))

            return mean(_acc_all)
