import sys
import copy

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.autograd import Variable

from data_set import AmazonDomainDataSet, AmazonSubsetWrapper
from helper.measure import acc
from torch.utils.data import DataLoader

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

    def fit(self, training_generator, validation_generator, target_data_set, dict):
        # target_generator = DataLoader(target_data_set, **{'batch_size': len(target_data_set)})
        # self._fit(training_generator, validation_generator, target_generator, self.max_epochs)
        # torch.save(self.model.state_dict(), 'tmp/model.pkl')
        self.model.load_state_dict(torch.load('tmp/model.pkl'))

        params = {'batch_size': 8,
                  'shuffle': False,
                  'num_workers': 1}

        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = dict

        # Pseudo labeling
        _tgt_len = len(target_data_set)

        for _i in range(1, 21):
            batch_num = 0
            wrong_target = 0

            _len = int((_i / 20) * _tgt_len)
            target_data_set.length = _len
            target_generator = DataLoader(target_data_set, **{'batch_size': _len, 'shuffle': True})
            with torch.set_grad_enabled(False):
                self.model.eval()
                for idx, batch_one_hot, labels, src in target_generator:
                    batch_num += 1
                    # Transfer to GPU
                    batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

                    f1, f2, ft = self.model(batch_one_hot)
                    t = Variable(torch.cuda.FloatTensor([0.5]))  # threshold
                    out1 = (f1 > t)
                    out2 = (f2 > t)
                    outt = (ft > t)

                    out = outt.cpu().numpy().flatten()
                    _idx = idx.cpu().numpy().flatten()
                    labels1 = labels.cpu().numpy()

                    for i in tqdm(range(0, len(outt))):
                        if out1[i][0] == out2[i][0]:
                            item = copy.deepcopy(target_generator.dataset.get(_idx[i]))
                            if item.sentiment != outt[i][0]:
                                wrong_target += 1
                            item.sentiment = np.int64(outt[i][0].cpu())
                            training_tgt_data_set.append(item, i)
                training_tgt_data_loader = DataLoader(training_tgt_data_set, **params)
                print('Wrong labeled: {} on {}'.format(wrong_target, _len))

            # Step 2
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=4)
            self._fit(training_tgt_data_loader, target_generator, max_epochs=20, is_step2=True)


    def _fit(self, data_loader: DataLoader, validation_loader: DataLoader, target_generator: DataLoader = None,
             max_epochs=10, is_step2=False):
        self.model.train()
        # Loop over epochs
        epochs_no_improve = 5
        n_epochs_stop = 0
        _prev_metric = 0

        for epoch in range(max_epochs):
            # Training
            batch_num = 0
            loss_all = 0
            loss_all1 = 0
            loss_all2 = 0
            _acc_all = 0
            _acc_all1 = 0
            _acc_all2 = 0
            for idx, batch_one_hot, labels, src in data_loader:
                batch_num += 1
                # Transfer to GPU
                batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

                # Model computations
                self.optimizer.zero_grad()
                # Forward pass
                f1, f2, ft = self.model(batch_one_hot)
                # Compute Loss
                # f1_out, f2_out, w1, w2, y, _del
                loss = self.criterion(f1, f2, self.model.f1_1.weight, self.model.f2_1.weight, labels, 0.01)
                loss_t = self.criterion_t(ft, labels)
                loss_all += loss.item()

                if is_step2:

                    _acc = acc(ft, labels)
                    _acc_all += _acc
                    _loss_mean = round(loss_all / batch_num, 10)

                    sys.stdout.write(
                        '\r### Ft: Epoch {}, Batch {}, train loss: {} , acc: {} ###'.format(epoch, batch_num,
                                                                                            _loss_mean,
                                                                                            round(_acc_all / batch_num,
                                                                                                  4)))
                    sys.stdout.flush()
                else:
                    _loss_mean = round(loss_all / batch_num, 4)

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
                                _acc_all1 / batch_num,
                                4),
                            round(
                                _acc_all2 / batch_num,
                                4)
                        ))

                    sys.stdout.flush()
                # Backward pass
                loss.backward(retain_graph=True)
              #  if not is_step2:
                loss_t.backward()
                self.optimizer.step()

            self.scheduler.step(_loss_mean, epoch)

            sys.stdout.write('\n')
            _valid_acc = self.valid(validation_loader, target_generator)

            if _valid_acc <= _prev_metric:
                n_epochs_stop += 1
                if n_epochs_stop == epochs_no_improve:
                    print('Early stopping!')
                    break
            else:
                _prev_metric = _valid_acc


    def valid(self, data_loader, data_loader_2=None):
        with torch.set_grad_enabled(False):
            self.model.eval()
            for idx, local_batch, local_labels, src in data_loader:
                local_batch = local_batch.to(device, torch.float)
                f1, f2, ft = self.model(local_batch)
                _acc = round(acc(ft, local_labels) * 100, 2)
            if data_loader_2 is not None:
                for idx, local_batch, local_labels, src in data_loader_2:
                    local_batch = local_batch.to(device, torch.float)
                    f1, f2, ft = self.model(local_batch)
                    _acc_tgt = round(acc(ft, local_labels) * 100, 2)
                print('\r acc SOURCE_VALID: {}%, acc TARGET: {}%\n'.format(_acc, _acc_tgt))
            else:
                print('\r acc SOURCE_VALID: {}%'.format(_acc))

            return _acc
