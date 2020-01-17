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
from nn.loss import KLDivergenceLoss
from nn.model import ModelWithTemperature, ATTFeedforward, SimpleAutoEncoder
from utils.measure import acc
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class AEGeneratorTrainer:
    def __init__(self, model, reconstruction_criterion, discrimination_criterion, generator_criterion, g_optimizer, d_optimizer, ae_optimizer, scheduler, max_epochs, epochs_no_improve=3):
        self.model = model
        self.reconstruction_criterion = reconstruction_criterion
        self.discrimination_criterion = discrimination_criterion
        self.generator_criterion = generator_criterion
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.ae_optimizer = ae_optimizer
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.epochs_no_improve = epochs_no_improve

    def fit(self, src_data_generator, tgt_data_generator):
        i_epoch = 0
        n_batches = len(src_data_generator)
        for epoch in range(self.max_epochs):
            self.model.train()
            i_batch = 0
            i_epoch += 1
            generator_losses = [0]
            reconstruction_losses = [0]
            domain_losses = [0]
            tgt_iter = iter(tgt_data_generator)

            d_steps = 1
            g_steps = 1
            r_steps = 1
            for _, inputs, y_inputs, domain_tg in src_data_generator:
                self.ae_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                i_batch += 1
                _, tgt_inputs, tgt_labels, domain_tg_tgt = next(tgt_iter)

                if type(y_inputs) == list:
                    y_inputs = torch.stack(y_inputs, dim=1)
                if type(domain_tg) == list:
                    domain_tg = torch.stack(domain_tg, dim=1)
                if type(domain_tg_tgt) == list:
                    domain_tg_tgt = torch.stack(domain_tg_tgt, dim=1)
                inputs, tgt_inputs, tgt_labels, y_inputs, domain_tg, domain_tg_tgt = inputs.to(device, torch.float), tgt_inputs.to(device, torch.float), tgt_labels.to(device, torch.float), y_inputs.to(device, torch.float), domain_tg.to(device, torch.float), domain_tg_tgt.to(device, torch.float)

                src_reconstructed_x, domain_class = self.model(inputs)
                tgt_reconstructed_x, tgt_domain_class = self.model(tgt_inputs)

                # Reconstruction
                src_loss = self.reconstruction_criterion(src_reconstructed_x, y_inputs)
                tgt_loss = self.reconstruction_criterion(tgt_reconstructed_x, tgt_labels)

                loss = src_loss + tgt_loss
                reconstruction_losses.append(loss.item())
                loss.backward(retain_graph=True)
                self.ae_optimizer.step()

                if epoch > r_steps:
                    # The Discriminator learning
                    for _ in range(d_steps):
                        self.d_optimizer.zero_grad()

                        src_loss = self.discrimination_criterion(torch.squeeze(domain_class), domain_tg)
                        tgt_loss = self.discrimination_criterion(torch.squeeze(tgt_domain_class), domain_tg_tgt)

                        loss = (src_loss + tgt_loss) / 2
                        loss.backward()

                        self.d_optimizer.step()

                        domain_losses.append(loss.item())

                    # The Generator Learning
                    for _ in range(g_steps):
                        self.g_optimizer.zero_grad()

                        _, tgt_inputs, tgt_labels, domain_tg_tgt = next(tgt_iter)

                        if type(y_inputs) == list:
                            y_inputs = torch.stack(y_inputs, dim=1)
                        if type(domain_tg) == list:
                            domain_tg = torch.stack(domain_tg, dim=1)
                        if type(domain_tg_tgt) == list:
                            domain_tg_tgt = torch.stack(domain_tg_tgt, dim=1)
                        inputs, tgt_inputs, tgt_labels, y_inputs, domain_tg, domain_tg_tgt = inputs.to(device,
                                                                                                       torch.float), tgt_inputs.to(
                            device, torch.float), tgt_labels.to(device, torch.float), y_inputs.to(device,
                                                                                                  torch.float), domain_tg.to(
                            device, torch.float), domain_tg_tgt.to(device, torch.float)

                        _, tgt_domain_class = self.model(tgt_inputs)

                        g_tgt_loss = self.discrimination_criterion(torch.squeeze(tgt_domain_class), domain_tg) # Train G to pretend it's genuine
                        g_tgt_loss.backward()
                        self.g_optimizer.step()

                        generator_losses.append(g_tgt_loss.item())

                sys.stdout.write('\r+\tepoch: %d / %d, batch: %d / %d, %s: %4.4f, %4.4f, %4.4f' % (i_epoch, self.max_epochs, i_batch, n_batches, 'BCE: ', mean(reconstruction_losses), mean(domain_losses), mean(generator_losses)))

            self.scheduler.step(mean(reconstruction_losses))
            print('')

            # acces = []
            # self.model.eval()
            # for _, inputs, y_inputs, sentiment in tgt_data_generator:
            #     if type(y_inputs) == list:
            #         y_inputs = torch.stack(y_inputs, dim=1)
            #     if type(sentiment) == list:
            #         sentiment = torch.stack(sentiment, dim=1)
            #
            #     inputs, y_inputs, sentiment = inputs.to(device, torch.float), y_inputs.to(device, torch.float), sentiment.to(device, torch.float)
            #
            #     reconstructed_x, label_class = self.model(inputs)
            #
            #     label_class = torch.squeeze(label_class)
            #     sentiment = torch.max(sentiment, 1)[1].to(dtype=torch.float)
            #     acces.append(self.valid(label_class, sentiment))
            #
            # print('TGT Acc: %4.4f' % mean(acces))

    def valid(self, output, ground_truth):

        return acc(output, ground_truth)


class AutoEncoderTrainer:
    def __init__(self, model, criterion, optimizer, optimizer_kl, scheduler, max_epochs, epochs_no_improve=3):
        self.model: SimpleAutoEncoder = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_kl = optimizer_kl
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.epochs_no_improve = epochs_no_improve

    def fit(self, train_data_generator, tgt_data_generator):
        print("> Training is running...")
        self.model.train()
        self.model.set_train_mode(True)

        # criterion = torch.nn.BCEWithLogitsLoss()
        # criterion_domain = torch.nn.BCEWithLogitsLoss()

        batches_src = [None] * len(train_data_generator)
        batches_tgt = [None] * len(tgt_data_generator)

        prev_loss = 999
        prev_model = None

        for epoch in range(self.max_epochs):
            for i, d in enumerate(train_data_generator):
                batches_src[i] = d

            for i, d in enumerate(tgt_data_generator):
                batches_tgt[i] = d

            self.model.set_train_mode(True)
            _loss = []
            _loss_domain = []
            _batch = 0
            tgt_data_iter = iter(tgt_data_generator)
            batches_n = math.ceil(len(train_data_generator.dataset) / train_data_generator.batch_size)
            print('+ \tepoch number: ' + str(epoch))
            counter = 0
            src_batches = []
            tgt_batches = []
            for idx, inputs, labels, domain_gt in random.sample(batches_src + batches_tgt, len(batches_src + batches_tgt)):
                #tgt_idx, tgt_inputs, tgt_labels, tgt_domain_gt = next(tgt_data_iter)
                _batch += 1
                # if type(labels) == list:
                #     labels = torch.stack(labels, dim=1)
                # if type(tgt_labels) == list:
                #     tgt_labels = torch.stack(tgt_labels, dim=1)

                inputs, labels = inputs.to(device, torch.float), labels.to(device, torch.float)
                # tgt_inputs, tgt_labels = tgt_inputs.to(device, torch.float), tgt_labels.to(device, torch.float)

                self.optimizer.zero_grad()

                # out = F.sigmoid(self.model(inputs))
                # tgt_out = F.sigmoid(self.model(tgt_inputs))

                out, src_encoded = self.model(inputs)
                # tgt_out, tgt_encoded = self.model(tgt_inputs)
                # out = F.sigmoid(out)
                # tgt_out = F.sigmoid(tgt_out)
                #self.model.unfroze()

                loss = self.criterion(out, labels)
                # loss = self.criterion(out, labels)
                # tgt_loss = self.criterion(tgt_out, tgt_labels)
                _loss.append(loss.item())
                #_loss.append(tgt_loss.item())

                # loss += tgt_loss
                loss.backward()
                self.optimizer.step()

                # loss_kl = kl_criterion()

                # loss_domain = criterion_domain(torch.squeeze(domain), torch.squeeze(domain_gt))
                # _loss_domain.append(loss_domain.item())
                #
                # loss_domain.backward()
                # self.optimizer.step()
                #
                # p = float(_batch) / batches_n
                # lambd = 2. / (1. + np.exp(-10. * p)) - 1
                # self.model.lambd = lambd

                _loss_mean = round(mean(_loss), 5)
                # _loss_mean_domain = round(mean(_loss_domain), 5)
                sys.stdout.write(
                    '\r+\tbatch: {}, {}: {}'.format(_batch, self.criterion.__class__.__name__,
                                                         _loss_mean))

                counter += 1

                #src_batches.append(inputs)
               # tgt_batches.append(tgt_inputs)

            # src_batches = torch.cat(src_batches, dim=0)
            # tgt_batches = torch.cat(tgt_batches, dim=0)

            self.model.set_train_mode(False)

            # with torch.no_grad():
            #     src_encoded = self.model(src_batches)

            # print('\n')
            # for _ in range(20):
            #     tgt_encoded = self.model(tgt_batches)
            #
            #     loss = KLDivergenceLoss()(src_encoded, tgt_encoded)
            #     loss.backward()
            #
            #     print(loss.item())
            #     self.optimizer_kl.step()

            if prev_loss <= _loss_mean:
                n_epochs_stop += 1
            else:
                prev_model = self.model
                n_epochs_stop = 0
                prev_loss = _loss_mean

            if n_epochs_stop == self.epochs_no_improve:
                self.model = prev_model
                print('Early Stopping!')
                break

            print('')
            self.scheduler.step(mean(_loss))

            batches_src = [None] * len(train_data_generator)
            batches_tgt = [None] * len(tgt_data_generator)
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

    def fit(self, training_loader: DataLoader, validation_loader: DataLoader, additional_training_data_set=None,
            target_generator=None, max_epochs=10, is_step2=False, _dict=None, calibrate=False):
        n_epochs_stop = 0
        prev_metric = 0

        target_generator = DataLoader(target_generator.dataset, shuffle=True, batch_size=target_generator.batch_size)
        for epoch in range(max_epochs):
            self.model.train()
            print('+ \tepoch number: ' + str(epoch))
            n_batch = 0
            loss_f1f2 = []
            acc_all = []
            batches = []
            # Init training data
            for idx, batch_one_hot, labels, src in training_loader:
                batches.append((idx, batch_one_hot, labels, src, True))

            if additional_training_data_set is not None:
                training_tgt_generator = additional_training_data_set
                for idx, batch_one_hot, labels, src in training_tgt_generator:
                    batches.append((idx, batch_one_hot, labels, src, False))

            random.shuffle(batches)

            # print(len(training_loader))
            # print(len(target_generator))
            # Domain divergence
            #src_iter = iter(training_loader)
            tgt_iter = iter(target_generator)

            for idx, input, labels, src, loss_upd in batches:
                n_batch += 1

                try:
                    idx_tgt, input_tgt, labels_tgt, src_tgt = next(tgt_iter)
                except:
                   # tgt_iter = iter(target_generator)
                    idx_tgt, input_tgt, labels_tgt, src_tgt = next(tgt_iter)

                if type(labels) == list:
                    labels = torch.stack(labels, dim=1)
                if type(labels_tgt) == list:
                    labels_tgt = torch.stack(labels_tgt, dim=1)


                # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices
                # max(1) will return the maximal value (and index in PyTorch) in this particular dimension.
                input, labels_, src = input.to(device, torch.float), torch.max(labels, 1)[1].to(device, torch.long), src.to(device, torch.float)
                input_tgt, labels_tgt, src_tgt = input_tgt.to(device, torch.float), torch.max(labels_tgt, 1)[1].to(device, torch.long), src_tgt.to(device, torch.float)
                # if self.ae_model is not None:
                #     input = self.ae_model(input)
                    #input = torch.cat([input, ae_output], 1)

                self.optimizer.zero_grad()

                # AutoEncoder domain discrepancy
                # _, src_batch_data, _, _ = next(src_iter)

                self.ae_model.zero_grad()

                enc_src_out = self.ae_model(input)
                enc_tgt_out = self.ae_model(input_tgt)
                #################################

                f1, f2, ft, _ = self.model(input)
                _loss = self.criterion(f1, f2, self.model.f1_1.weight, self.model.f2_1.weight, enc_src_out, enc_tgt_out, labels_)

          #      loss_rev = F.binary_cross_entropy_with_logits(torch.squeeze(rev), src)

                if not is_step2 or not loss_upd:
                    _loss_t = self.criterion_t(ft, labels_)
                    _loss = _loss + _loss_t

                # if is_step2:
                    # _loss = _loss + _loss_rev #

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

            print('> Validation data acc: ')
            _valid_acc = self.valid(validation_loader, valid=True)

            if target_generator is not None:
                print('> Target data acc: ')
                self.valid(target_generator, valid=False)

            self.scheduler.step(_valid_acc)

            if _valid_acc <= prev_metric:
                n_epochs_stop += 1
                if n_epochs_stop == self.epochs_no_improve:
                    # self.model = prev_model
                    print('Early stopping!')
                    break
            else:
                prev_model = self.model
                n_epochs_stop = 0
                prev_metric = _valid_acc

        # calibrating
        # if calibrate:
        #     self.calibrated_model = ModelWithTemperature(self.model, self.ae_model)
        #     self.calibrated_model.set_temperature(validation_loader)
        # else:
        #     self.calibrated_model = self.model


    def valid(self, data_loader, data_loader_2=None, valid=False):
        with torch.set_grad_enabled(False):
            self.model.eval()
            _acc_all = []
            _acc_all_tgt = []
            for idx, input, local_labels, src in data_loader:
                local_labels = torch.stack(local_labels, dim=1)
                input = input.to(device, torch.float)
                # if self.ae_model:
                #     input = self.ae_model(input)
                   # input = torch.cat([input, ae_output], 1)
                f1, f2, ft, rev = self.model(input)
                if valid:
                    _acc_all.append(
                        (acc(F.softmax(f1, dim=1), local_labels) + acc(F.softmax(f2, dim=1), local_labels)) / 2)
                else:
                    _acc_all.append(acc(F.softmax(ft, dim=1), local_labels))
            if data_loader_2 is not None:
                for idx, input, local_labels, src in data_loader_2:
                    local_labels = torch.stack(local_labels, dim=1)
                    input = input.to(device, torch.float)
                    # if self.ae_model:
                    #     input = self.ae_model(input)
                    # input = torch.cat([input, ae_output], 1)
                    f1, f2, ft, rev = self.model(input)
                    if valid:
                        _acc_all_tgt.append(
                            (acc(F.softmax(f1, dim=1), local_labels) + acc(F.softmax(f2, dim=1), local_labels)) / 2)
                    else:
                        _acc_all_tgt.append(acc(F.softmax(ft, dim=1), local_labels))
                print('\r+\tacc valid: {}, acc target: {}\n'.format(round(mean(_acc_all), 4),
                                                                    round(mean(_acc_all_tgt), 4)))
            else:
                print('\r+\tacc valid: {}'.format(round(mean(_acc_all), 4)))

            return mean(_acc_all)

    def pseudo_label(self, training_generator: DataLoader, valid_generator: DataLoader, target_data_set, train_params,
                     iterations=20, max_epochs=3):
        # print(target_data_set.data)
        target_generator = DataLoader(target_data_set, batch_size=len(target_data_set))

        # src_generator = DataLoader(src_data_set, **train_params)
        idx_added = []
        training_tgt_data_set = AmazonDomainDataSet()
        training_tgt_data_set.dict = target_data_set.dict
        n_all_wrong_labeled = 0
        n_all = 0

        for _iter in range(iterations):
            # valid_data = AmazonDomainDataSet()
            # valid_data.dict = target_generator.dataset.dict
            # n_examples_per_iteration = int(training_generator.dataset.length / iterations)
            # training_generator.dataset.length = n_examples_per_iteration * (iterations - _iter)
            # training_generator = DataLoader(training_generator.dataset, shuffle=True, batch_size=training_generator.batch_size)
            n_wrong_labeled = 0
            n_all_qualified = 0
            LIMIT = int(len(target_generator.dataset) / iterations)

            idx, f1, f2, fn = self._predict(target_generator)
            f1_max, f1_idx = torch.max(f1, dim=1)
            f2_max, f2_idx = torch.max(f2, dim=1)
            fn_max, fn_idx = torch.max(fn, dim=1)

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

            # for _i in most_confident_idxs:
            #     if f1_idx[_i] == f2_idx[_i]:
            #         idx_added.append(_i)
            #         item = copy.deepcopy(target_generator.dataset.get(_i))
            #         # if np.argmax(item.sentiment) != np.argmax(f1[_i].cpu().numpy()):
            #         #     n_wrong_labeled += 1
            #         item.sentiment = (1, 0) if f1_idx[_i] == 0 else (0, 1)
            #         # training_tgt_data_set.append(item)
            #         valid_data.append(item)

            for _i in most_confident_idxs:
                if f1_idx[_i] == f2_idx[_i] and f1_max[_i] > 0.90 and f2_max[_i] > 0.90:
                    n_all_qualified += 1
                    idx_added.append(_i)
                    item = copy.deepcopy(target_generator.dataset.get(_i))
                    if np.argmax(item.sentiment) != np.argmax(f1[_i].cpu().numpy()):
                        n_wrong_labeled += 1
                    item.sentiment = (1, 0) if f1_idx[_i] == 0 else (0, 1)
                    training_tgt_data_set.append(item)
                else:
                    pass

            for _i, _ in idxs:
                if _i not in idx_added:
                    item = copy.deepcopy(target_generator.dataset.get(_i))
                    item.sentiment = (1, 0) if f1_idx[_i] == 0 else (0, 1)
                    # valid_data.append(item)

            n_all_wrong_labeled += n_wrong_labeled
            n_all += n_all_qualified
            print('Iteration: {}, wrong labeled count: {} on {} qualified, all {}. All wrong labeled: {} / {} [{}%]'.format(_iter, n_wrong_labeled, n_all_qualified, len(idx), n_all_wrong_labeled, n_all, round((n_all_wrong_labeled/ n_all), 4) * 100))

            # train_generator, valid_generator, target_generator = train_valid_target_split(training_tgt_data_set,
            #                                                                             target_data_set,
            #                                                                           train_params)

            self.model.reset()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=2)

            # train_tgt_generator, valid_tgt_generator, _ = train_valid_target_split(training_tgt_data_set, training_tgt_data_set, train_params)

            # valid_generator = DataLoader(valid_data, shuffle=True, batch_size=len(valid_data))
            # src_generator.dataset.length = (iterations - (_iter + 1)) * int((src_generator.dataset.length / iterations))
            self.fit(training_generator, valid_generator,
                     additional_training_data_set=DataLoader(training_tgt_data_set, shuffle=True,
                                                             batch_size=training_generator.batch_size),
                     target_generator=target_generator,
                     is_step2=True,
                     max_epochs=6,
                     calibrate=False)


    def _predict(self, data_generator: DataLoader):
        assert data_generator.batch_size == len(data_generator.dataset)
        self.model.eval()
        with torch.set_grad_enabled(False):
            for idx, input, labels, src in data_generator:
                input = input.to(device, torch.float)
                # if self.ae_model:
                #     input = self.ae_model(input)
                # input = torch.cat([input, ae_output], 1)
                f1, f2, ft, rev = self.model(input)
                return idx, F.softmax(f1, dim=1), F.softmax(f2, dim=1), F.softmax(ft, dim=1)
