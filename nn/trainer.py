import sys
import copy

import torch
import numpy as np
from tqdm import tqdm
from helper.data import doc2one_hot
from torch.autograd import Variable
from helper.measure import acc
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# torch.cuda.set_device(device)


class DomainAdaptationTrainer:

    def __init__(self, model, criterion, criterion_t, optimizer, max_epochs):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.model.to(device)

    def fit(self, training_generator, validation_generator, target_generator):
        self._fit(training_generator, validation_generator, target_generator, self.max_epochs)
        torch.save(self.model.state_dict(), 'tmp/model.pkl')
        # self.model.load_state_dict(torch.load('tmp/model.pkl'))

        # pseudo labeling
        batch_num = 0
        wrong_target = 0
        for idx, batch_one_hot, labels in target_generator:
            batch_num += 1
            # Transfer to GPU
            batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

            f1, f2, ft = self.model(batch_one_hot)
            t = Variable(torch.FloatTensor([0.5]))  # threshold
            out1 = (f1 > t)
            out2 = (f2 > t)

            for i in tqdm(range(0, len(out1))):
                if out1[i] == out2[i]:
                    item = copy.deepcopy(target_generator.dataset.get(i))
                    if item.sentiment != out1[i][0]:
                        wrong_target += 1
                    item.sentiment = np.int64(out1[i][0])
                    training_generator.dataset.append(item)
        print('Wrong labeled: {} on {}'.format(wrong_target, len(target_generator)))
        # Step 2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._fit(training_generator, target_generator, max_epochs=5)

    def _fit(self, data_loader: DataLoader, validation_loader: DataLoader, target_generator: DataLoader = None, max_epochs=10):
        # Loop over epochs

        for epoch in range(max_epochs):
            # Training
            batch_num = 0
            loss_all = 0
            _acc_all = 0
            for idx, batch_one_hot, labels in data_loader:
                batch_num += 1
                # Transfer to GPU
                batch_one_hot, labels = batch_one_hot.to(device, torch.float), labels.to(device, torch.float)

                # Model computations
                self.optimizer.zero_grad()
                # Forward pass
                f1, f2, ft = self.model(batch_one_hot)
                # Compute Loss
                # f1_out, f2_out, w1, w2, y, _del
                loss = self.criterion(f1, f2, ft, self.model.f1_1.weight, self.model.f2_1.weight, labels, 0.01)
                loss_t = self.criterion_t(ft, labels)
                loss_all += loss.item()

                _acc = acc(ft, labels)
                _acc_all += _acc

                sys.stdout.write('\r### Epoch {}, Batch {}, train loss: {} , acc: {} ###'.format(epoch, batch_num,
                                                                                       round(loss_all / batch_num, 4),
                                                                                       round(_acc_all / batch_num, 4)))
                sys.stdout.flush()
                # Backward pass
                loss.backward(retain_graph=True)
                loss_t.backward()
                self.optimizer.step()

            sys.stdout.write('\n')
            self.valid(validation_loader, target_generator)

    def valid(self, data_loader, data_loader_2=None):
        with torch.set_grad_enabled(False):
            for idx, local_batch, local_labels in data_loader:
                local_batch = local_batch.to(device, torch.float)
                f1, f2, ft = self.model(local_batch)
                _acc = round(acc(ft, local_labels) * 100, 2)
            if data_loader_2 is not None:
                for idx, local_batch, local_labels in data_loader_2:
                    local_batch = local_batch.to(device, torch.float)
                    f1, f2, ft = self.model(local_batch)
                    _acc_tgt = round(acc(ft, local_labels) * 100, 2)
                print('\r acc SOURCE_VALID: {}%, acc TARGET: {}%\n'.format(_acc, _acc_tgt))
            else:
                print('\r acc SOURCE_VALID: {}%'.format(_acc))