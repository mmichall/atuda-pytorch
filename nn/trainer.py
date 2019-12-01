import sys
import copy
from statistics import mean
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.autograd import Variable

from data_set import AmazonDomainDataSet, AmazonSubsetWrapper
from helper.measure import acc
from torch.utils.data import DataLoader

from helper.data import doc2one_hot

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# torch.cuda.set_device(device)

class DomainAdaptationTrainer:

    def __init__(self, model, criterion, criterion_t, optimizer, scheduler, max_epochs):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.model.to(device)

    def fit(self, training_generator, validation_generator, target_data_set, _dict):
        # target_generator = DataLoader(target_data_set, **{'batch_size': len(target_data_set)})
        # self._fit(training_generator, validation_generator, target_generator, self.max_epochs, _dict=_dict)
        # torch.save(self.model.state_dict(), 'tmp/model.pkl')
        self.model.load_state_dict(torch.load('tmp/model.pkl'))

        params = {'batch_size': 8,
                  'shuffle': False,
                  'num_workers': 1}

        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = _dict

        # Pseudo labeling
        _tgt_len = len(target_data_set)

        wrong_target = 0
        idxs_to_remove = []
        for _i in range(1, 16):
            batch_num = 0
            _len = int((_i / 15) * _tgt_len)
            target_data_set.length = _tgt_len
            target_generator = DataLoader(target_data_set, **{'batch_size': _tgt_len, 'shuffle': True})
            with torch.set_grad_enabled(False):
                self.model.eval()
                for idx, batch_one_hot, labels in target_generator:
                    batch_num += 1

                    # Transfer to GPU
                    labels = torch.stack(labels, dim=1)
                    batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

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
                        if np.array_equal(out1[i].cpu().numpy(), out2[i].cpu().numpy())\
                               and max(f1[i]) > 0.95 \
                               and max(f2[i]) > 0.95:
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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=3)
        self._fit(training_tgt_data_loader, target_generator, max_epochs=20, is_step2=True)


    def _fit(self, data_loader: DataLoader, validation_loader: DataLoader, target_generator: DataLoader = None,
             max_epochs=10, src_data_loader: DataLoader = None, is_step2=False, _dict=None):
        self.model.train()
        # Loop over epochs
        epochs_no_improve = 4
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

                # Model computations
                self.optimizer.zero_grad()
                # Forward pass
                f1, f2, ft = self.model(batch_one_hot)
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
                    _acc_all1 += _acc1

                    _acc2 = acc(f2, labels)
                    _acc_all2 += _acc2

                    sys.stdout.write(
                        '\r### Epoch {}, Batch {}, train loss: {} F1 acc: {} ### | ### F2 acc: {} ###'.format(
                            epoch,
                            batch_num,
                            _loss_mean,
                            round(
                                mean(_acc_all1),
                                4),
                            round(
                                mean(_acc_all2),
                                4)
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
                f1, f2, ft = self.model(local_batch)
                _acc_all.append(acc(ft, local_labels))
            if data_loader_2 is not None:
                for idx, local_batch, local_labels in data_loader_2:
                    local_labels = torch.stack(local_labels, dim=1)
                    local_batch = local_batch.to(device, torch.float)
                    f1, f2, ft = self.model(local_batch)
                    _acc_all_tgt.append(acc(ft, local_labels))
                print('\r acc SOURCE_VALID: {}, acc TARGET: {}\n'.format(mean(_acc_all), mean(_acc_all_tgt)))
            else:
                print('\r acc SOURCE_VALID: {}'.format(mean(_acc_all)))

            return mean(_acc_all)
