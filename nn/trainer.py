import random
import sys
import copy
from statistics import mean
import operator
import torch
import numpy as np
from sklearn.svm import LinearSVC
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from data_set import AmazonDomainDataSet, merge, AmazonSubsetWrapper
from nn.loss import KLDivergenceLoss, DLoss
from nn.model import SimpleAutoEncoder, DistNet, LogisticRegression, Discriminator
from utils.data import train_valid_split
from utils.measure import acc
from torch.utils.data import DataLoader
from sklearn import utils

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class AEGeneratorTrainer:
    def __init__(self, model, reconstruction_criterion, discrimination_criterion, generator_criterion, g_optimizer,
                 d_optimizer, ae_optimizer, scheduler, max_epochs, epochs_no_improve=3):
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
                inputs, tgt_inputs, tgt_labels, y_inputs, domain_tg, domain_tg_tgt = inputs.to(device,
                                                                                               torch.float), tgt_inputs.to(
                    device, torch.float), tgt_labels.to(device, torch.float), y_inputs.to(device,
                                                                                          torch.float), domain_tg.to(
                    device, torch.float), domain_tg_tgt.to(device, torch.float)

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

                        g_tgt_loss = self.discrimination_criterion(torch.squeeze(tgt_domain_class),
                                                                   domain_tg)  # Train G to pretend it's genuine
                        g_tgt_loss.backward()
                        self.g_optimizer.step()

                        generator_losses.append(g_tgt_loss.item())

                sys.stdout.write('\r+\tepoch: %d / %d, batch: %d / %d, %s: %4.4f, %4.4f, %4.4f' % (
                    i_epoch, self.max_epochs, i_batch, n_batches, 'BCE: ', mean(reconstruction_losses),
                    mean(domain_losses),
                    mean(generator_losses)))

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
    def __init__(self, src_domain, tgt_domain, model, criterion, optimizer, optimizer_kl, scheduler, max_epochs,
                 epochs_no_improve=3,
                 model_file='', kl_threshold=0):
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        self.model: SimpleAutoEncoder = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=0.1)
        self.optimizer_kl = optimizer_kl
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.epochs_no_improve = epochs_no_improve
        self.model_file = model_file
        self.kl_threshold = kl_threshold
        self.discriminator = Discriminator(self.model.shape[0]).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)
        self.optimizer_2 = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.discriminator_criterion = BCELoss()

    def fit(self, train_data, tgt_data, shuffle=True, denoising_factor=0.0, batch_size=8, return_input=True):
        print("> Training is running...")
        self.model.train()
        self.model.set_train_mode(True)

        data_set = merge([train_data, tgt_data])
        train_idxs, valid_idxs = train_valid_split(0, len(data_set), 0.2)
        print(len(train_data))
        print(len(tgt_data))
        print(len(data_set))
        data_set.dict.update(train_data.dict)
        data_set.dict.update(tgt_data.dict)
        data_set.denoising_factor = denoising_factor
        data_set.return_input = return_input
        data_set.summary('Training data set')

        print("Training set length: {}, Validation set length: {}".format(len(train_idxs), len(valid_idxs)))

        train_subset = AmazonSubsetWrapper(data_set, train_idxs)
        valid_subset = AmazonSubsetWrapper(data_set, valid_idxs)

        data_generator = DataLoader(data_set, shuffle=shuffle, batch_size=batch_size)
        train_data_generator = DataLoader(train_subset, shuffle=shuffle, batch_size=batch_size)
        valid_data_generator = DataLoader(valid_subset, shuffle=shuffle, batch_size=len(valid_subset))

        prev_loss = 999
        prev_model = None

        scheduler_1 = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=4)

        # lr_criterion = KLDivergenceLoss()
        # lr_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        #
        # lr = LogisticRegression(3000, 1)

        true_labels = torch.ones(batch_size, device=device)
        false_labels = torch.zeros(batch_size, device=device)

        with open('tmp/autoencoder_metrics.txt', 'a+') as f:

            # Train a simple target/source domain logistic regression classifier C
            n_epochs_stop_1 = 0
            acc_prev = 0
            for epoch in range(1):
                print('\nEpoch: ' + str(epoch))
                _batch = 0
                loss_mean = []

                for idx, inputs, labels, domain_gt, sentiment in train_data_generator:
                    if len(idx) < batch_size:
                        continue

                    self.discriminator_optimizer.zero_grad()
                    _batch += 1
                    # labels as real data
                    labels, domain_gt = labels.to(device, torch.float), domain_gt.to(device, torch.float)

                    domain_ind_out = torch.squeeze(self.discriminator(labels))
                    discriminator_loss = self.discriminator_criterion(domain_ind_out, domain_gt)
                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()
                    loss_mean.append(discriminator_loss.item())
                    sys.stdout.write(
                        '\r+\tbatch: {}, {}: {}'.format(_batch, self.discriminator_criterion.__class__.__name__,
                                                        round(discriminator_loss.item(), 5)))

                # scheduler_1.step(mean(loss_mean))

                for idx, inputs, labels, domain_gt, sentiment in valid_data_generator:
                    labels, domain_gt = labels.to(device, torch.float), domain_gt.to(device, torch.float)

                    out = self.discriminator(labels)
                    out = out.cpu().detach().numpy()
                    out = [1 if o >= 0.5 else 0 for o in out]
                    domain_gt = domain_gt.cpu().detach().numpy()
                    _, _, acc = measure(out, domain_gt)

                print(n_epochs_stop_1, acc_prev, acc)

                if acc_prev >= acc:
                    n_epochs_stop_1 += 1
                else:
                    n_epochs_stop_1 = 0
                    acc_prev = acc

                if n_epochs_stop_1 == 5:
                    print('Early Stopping!')
                    break

            _kl_losses = []
            _losses = []
            for epoch in range(self.max_epochs):
                self.model.train()
                self.model.set_train_mode(True)
                _loss = []
                _loss_domain = []
                _batch = 0
                _kl_losses_epoch = []
                _discriminator_losses = []
                print('+ \tepoch number: ' + str(epoch))
                counter = 0

                    ## Every each epoch ##
                    # data_generator_kl = DataLoader(data_set, shuffle=True, batch_size=len(data_set))
                    # for idx, inputs, labels, domain_gt, sentiment in data_generator_kl:
                    #     if len(idx) < batch_size:
                    #         continue
                    #     inputs, labels = inputs.to(device, torch.float), labels.to(device, torch.float)
                    #     self.optimizer.zero_grad()
                    # # #
                    #     self.model.set_train_mode(False)
                    #     self.model.eval()
                    #
                    #     # with torch.no_grad():
                    #     encoded, src_class = self.model(inputs)
                    #
                    #     domain_gt = domain_gt.to(device, torch.float)
                    #     src_class = src_class.to(device, torch.float)
                    #
                    #     print(np.squeeze(src_class))
                    #     print(domain_gt)

                    # lr_loss = lr_criterion(np.squeeze(src_class), domain_gt)
                    # lr_loss = -lr_loss
                    # lr_loss.backward()
                    # lr_optimizer.step()
                    # print(lr_loss.item())

                    # X = encoded.detach().cpu().numpy()
                    # domain_gt = domain_gt.detach().cpu().numpy()
                    # self.cls = LinearSVC(max_iter=20000, dual=False)
                    #
                    # q = encoded[domain_gt == 1].detach().cpu().numpy()
                    # p = encoded[domain_gt == 0].detach().cpu().numpy()
                    #
                    # l = [*q[:1000], *p[:1000]]
                    # dt = [*np.ones(1000), *np.zeros(1000)]
                    # l, dt = utils.shuffle(l, dt)
                    # self.cls.fit(l, dt)
                    # out = self.cls.predict(X)
                    #
                    # acc = np.round((domain_gt == out).sum() / len(domain_gt), 3)
                    # print(acc)
                    #
                    #     src_mi = torch.sum(src_encoded, dim=0) / src_encoded.size()[0]
                    #     tgt_mi = torch.sum(tgt_encoded, dim=0) / tgt_encoded.size()[0]
                    #
                    #     l2Loss = torch.sqrt(torch.sum(torch.square(src_mi - tgt_mi)))
                    #     print(l2Loss.item())
                    #     _kl_losses_epoch.append(l2Loss.item())
                    #
                    #     # kl_loss = KLDivergenceLoss()(src_encoded, tgt_encoded)
                    #     # _kl_losses_epoch.append(kl_loss.item())
                    #     # # [ <------> KL after every epoch <------> ] #
                    #     self.optimizer_kl.zero_grad()
                    #     l2Loss.backward()
                    #     self.optimizer_kl.step()

                    # q = inputs[domain_gt == 1]
                    # p = inputs[domain_gt == 0]
                    # q = self.model.encoder(q)
                    # p = self.model.encoder(p)
                    #
                    # lr_optimizer.zero_grad()
                    # kl_out = logistic_classifier(q).to(device, torch.float)
                    # kl_out = torch.flatten(kl_out)
                    #
                    # lr_loss = lr_criterion(kl_out, sentiment)
                    # print(' LR: ' + str(lr_loss.item()))
                _batch = 0
                for idx, inputs, labels, domain_gt, sentiment in data_generator:
                    if len(idx) < batch_size:
                        continue
                    _batch += 1
                    inputs, labels, domain_gt = inputs.to(device, torch.float), labels.to(device, torch.float), domain_gt.to(device, torch.float)

                    self.model.train()
                    self.model.set_train_mode(True)

                    # zero the gradients on each iteration
                    self.optimizer.zero_grad()
                    # reconstruction loss
                    out, _ = self.model(inputs)
                    reconstruction_loss = self.criterion(out, labels)
                    reconstruction_loss.backward()
                    self.optimizer.step()

                   # self.optimizer.zero_grad()
                    # # src domain examples as generated data
                   # out, _ = self.model(inputs)
                    #generated_data = torch.sigmoid(generated_data)
                    # # target domain examples from batch as true data
                    # true_data = labels[domain_gt == 0]
                    # _, true_data = self.model(inputs[domain_gt == 0])
                    #true_data = torch.sigmoid(true_data)
                    # # out as generated data
                    # discriminator_out = torch.squeeze(self.discriminator(generated_data))
                    # d_loss = 0.001 * self.discriminator_criterion(discriminator_out, torch.ones(len(discriminator_out), device=device))
                    # d_loss.backward()
                    # optimize generator/ model

                    # Train discriminator on the true/generated data
                    #class_dist = self.discriminator(F.sigmoid(out))
                    # true_discriminator_loss = self.discriminator_criterion(latent_data_out, domain_gt)

                    # add .detach() here think about this
                    # generator_discriminator_out = torch.squeeze(self.discriminator(generated_data.detach()))
                    # generator_discriminator_loss = self.discriminator_criterion(generator_discriminator_out, torch.zeros(len(generated_data), device=device))
                    # discriminator_loss = true_discriminator_loss
                    # discriminator_loss.backward()
                    #
                    # self.encoder_optimizer.step()
                    # self.discriminator_optimizer.step()

                    #discriminator_loss = DLoss()(class_dist)
                    #discriminator_loss.backward()
                    #self.optimizer.step()

                    _loss.append(reconstruction_loss.item())
                    _kl_losses_epoch.append(discriminator_loss.item())
                    _discriminator_losses.append(0) #(d_loss.mean().item())

                    # domain_gt = domain_gt.to(device, torch.float)
                    # q = src_encoded[domain_gt > 0]
                    # p = src_encoded[domain_gt == 0]

                    # src_var, src_mi = torch.var_mean(q, unbiased=False)
                    # tgt_var, tgt_mi = torch.var_mean(p, unbiased=False)

                    # lda_loss = torch.divide(torch.square(src_mi - tgt_mi), torch.square(src_var) + torch.square(tgt_var))

                    #
                    # l2Loss = torch.sqrt(torch.sum(torch.square(src_mi - tgt_mi)))

                    # domain_gt = utils.shuffle(domain_gt)
                    # src_class = src_class.to(device, torch.float)

                    # lr_loss = lr_criterion(np.squeeze(src_class), domain_gt)
                    # print(' ' + str(lr_loss.item()))

                    # tgt_out, tgt_encoded = self.model(tgt_inputs)
                    # out = F.sigmoid(out)
                    # tgt_out = F.sigmoid(tgt_out)
                    # self.model.unfroze()

                    # lr_loss.backward()
                    # lr_optimizer.step()
                    #
                    # kl_loss = KLDivergenceLoss()(q, p)
                    # print(kl_loss.item())

                    # loss = self.criterion(out, labels)# + 0.1 * kl_loss

                    # loss = self.criterion(out, labels)
                    # tgt_loss = self.criterion(tgt_out, tgt_labels)
                   # _loss.append(loss.item())
                    # _loss.append(tgt_loss.item())

                    # loss += tgt_loss
                   # loss.backward()
                  #  self.optimizer.step()

                    # _loss_mean_domain = round(mean(_loss_domain), 5)
                    sys.stdout.write(
                        '\r+\tbatch: {}, {}: {} {}'.format(_batch, self.criterion.__class__.__name__, round(reconstruction_loss.item(), 5), round(discriminator_loss.item(), 5)))

                    # counter += 1
                    # q = inputs[domain_gt > 0]
                    # p = inputs[domain_gt == 0]


                    # self.model.set_train_mode(False)
                    # self.model.eval()
                    #
                    # # with torch.no_grad():
                    # src_encoded, _ = self.model(q)
                    # tgt_encoded, _ = self.model(p)
                    #
                    # print(src_encoded, tgt_encoded)

                    # src_mi = torch.sum(src_encoded, dim=0) / src_encoded.size()[0]
                    # tgt_mi = torch.sum(tgt_encoded, dim=0) / tgt_encoded.size()[0]

                    # l2Loss = torch.sqrt(torch.sum(torch.square(src_mi - tgt_mi)))
                    # kl_loss = KLDivergenceLoss()(src_encoded, tgt_encoded)
                    # _kl_losses_epoch.append(kl_loss.item())

                    # self.optimizer.zero_grad()
                    # self.optimizer_kl.zero_grad()
                    # loss = loss + 0.0001 * kl_loss
                    # loss.backward()
                    # self.optimizer.step()
                        # print(' -> KL_Loss: ' + str(round(kl_loss.item(), 5)))
                        #
                        # loss = loss + 0.01 * kl_loss

                        # print(self.kl_threshold)
                        # if kl_loss.item() > self.kl_threshold:
                        # if epoch % 10 == 0:
                        # if kl_loss.item() > 0.0001:
                        # [ <------> KL <------> ] #
                        # kl_loss.backward()
                        # self.optimizer_kl.step()
                        # self.optimizer_kl.zero_grad()
                  #  _kl_losses_epoch.append(kl_loss.item())

                _loss_mean = round(mean(_loss), 5)
                _kl_loss_mean_ = round(mean(_kl_losses_epoch), 8)
                _kl_losses.append(_kl_loss_mean_)
                _losses.append(_loss_mean)
                print('\n[ LOSS MEAN: ' + str(_loss_mean) + ' KL_LOSS_MEAN: ' + str(_kl_loss_mean_) + ' ]' + ' D_LOSS: ' + str(round(mean(_discriminator_losses), 5)))
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

            f.write(self.src_domain + ' --> ' + self.tgt_domain + ' ' + str(self.kl_threshold) + '\n')
            f.write('[' + ', '.join(map(str, _kl_losses)) + ']' + '\n')
            f.write('[' + ', '.join(map(str, _losses)) + ']' + '\n')
            f.write('\n')

        model_file_name = self.model_file.format(self.src_domain, self.tgt_domain, _loss_mean, epoch)
        torch.save(self.model.encoder.state_dict(), model_file_name)
        print('Model was saved in {} file.'.format(model_file_name))
        print("> Training is over. Thank you for your patience :).")


class DistNetTrainer:
    def __init__(self, src_domain, tgt_domain, encoder_path=None, max_epochs=500):
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        self.max_epochs = max_epochs
        self.epochs_no_improve = 4
        self.dict = {}

        self.distNet = DistNet()
        if encoder_path:
            self.distNet.encoder.load_state_dict(torch.load(encoder_path))

        self.optimizer = Adam(self.distNet.parameters(), lr=0.00005)
        self.criterion = BCEWithLogitsLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=3)



    def fit(self, src_data: AmazonDomainDataSet, tgt_data, shuffle=True, batch_size=8):
        print("> Training is running...")

        train_idxs, valid_idxs = train_valid_split(0, len(src_data), 0.75)
        print("Training set length: {}, Validation set length: {}".format(len(train_idxs), len(valid_idxs)))

        src_data.dict.update(src_data.dict)
        src_data.dict.update(tgt_data.dict)
        self.dict = src_data.dict
        src_data.return_input = False
        src_data.summary('Training data set')

        train_subset = AmazonSubsetWrapper(src_data, train_idxs)
        valid_subset = AmazonSubsetWrapper(src_data, valid_idxs)

        prev_loss = 999
        prev_model = None

        _kl_losses = []
        _losses = []
        for epoch in range(self.max_epochs):
            _loss = []
            _loss_domain = []
            _batch = 0
            _kl_losses_epoch = []
            print('+ \tepoch number: ' + str(epoch))
            counter = 0

            src_data_generator = DataLoader(src_data, shuffle=shuffle, batch_size=len(src_data.data))
            tgt_data_generator = DataLoader(tgt_data, shuffle=shuffle, batch_size=len(tgt_data.data))

            with torch.no_grad():
                for _, src_inputs, _, _ in src_data_generator:
                    for _, tgt_inputs, _, _ in tgt_data_generator:
                        src_encoded = self.distNet.encoder(src_inputs.to(device, torch.float))
                        tgt_encoded = self.distNet.encoder(tgt_inputs.to(device, torch.float))

                        src_mi = torch.sum(src_encoded, dim=0) / src_encoded.size()[0]
                        tgt_mi = torch.sum(tgt_encoded, dim=0) / tgt_encoded.size()[0]

                        kl_loss = torch.sqrt(torch.sum(torch.square(src_mi - tgt_mi)))

                        # kl_loss = KLDivergenceLoss()(src_encoded, tgt_encoded)
                        print(kl_loss.item())

                        # kl_loss.backward()
                        # self.optimizer.step()

            data_generator = DataLoader(train_subset, shuffle=shuffle, batch_size=batch_size)

            for _, tgt_inputs, _, _ in tgt_data_generator:
                for idx, inputs, labels, domain_gt in data_generator:
                    if len(idx) < batch_size:
                        continue

                    src_encoded = self.distNet.encoder(inputs.to(device, torch.float))
                    tgt_encoded = self.distNet.encoder(tgt_inputs.to(device, torch.float))

                    src_mi = torch.sum(src_encoded, dim=0) / src_encoded.size()[0]
                    tgt_mi = torch.sum(tgt_encoded, dim=0) / tgt_encoded.size()[0]

                    kl_loss = torch.sqrt(torch.sum(torch.square(src_mi - tgt_mi)))

                    _batch += 1
                    inputs, labels = inputs.to(device, torch.float), labels[0].to(device, torch.float)
                    self.optimizer.zero_grad()
                    out = self.distNet(inputs)
                    out = torch.sum(out, dim=1)
                    loss = self.criterion(out, labels) + 0.1 * kl_loss
                    _loss.append(loss.item())
                    sys.stdout.write(
                            '\r+\tbatch: {}, {}: {}'.format(_batch, self.criterion.__class__.__name__, round(loss.item(), 10)))
                    counter += 1
                    _kl_losses_epoch.append(0)

                    self.optimizer.zero_grad()
                    loss = loss
                    loss.backward()
                    self.optimizer.step()

            print('\n')
            self.eval(valid_subset)
            self.eval(tgt_data)

            _loss_mean = round(mean(_loss), 10)
            _kl_loss_mean_ = round(mean(_kl_losses_epoch), 8)
            _kl_losses.append(_kl_loss_mean_)
            _losses.append(_loss_mean)
            print('\n[ LOSS MEAN: ' + str(_loss_mean) + ' KL_LOSS_MEAN: ' + str(_kl_loss_mean_) + ' ]')
            if prev_loss <= _loss_mean:
                n_epochs_stop += 1
            else:
                prev_model = self.distNet
                n_epochs_stop = 0
                prev_loss = _loss_mean

            if n_epochs_stop == self.epochs_no_improve:
                self.distNet = prev_model
                print('Early Stopping!')
                break

            print('')
            self.scheduler.step(mean(_loss))

        print("> Training is over. Thank you for your patience :).")


    def eval(self, tgt_data, shuffle=True):
        data_generator = DataLoader(tgt_data, shuffle=shuffle, batch_size=len(tgt_data))

        _kl_losses = []
        _losses = []

        for idx, inputs, labels, domain_gt in data_generator:
            inputs, labels = inputs.to(device, torch.float), labels[0].to(device, torch.float)
            out = torch.sigmoid(self.distNet(inputs)) > 0.5
            predicted = torch.sum(out, dim=1).cpu().detach().numpy()
            ground_truth = labels.cpu().detach().numpy()

            non_zeros, zeros, acc = np.count_nonzero(ground_truth), ground_truth.size - np.count_nonzero(ground_truth), \
                                    np.round((ground_truth == predicted).sum() / len(ground_truth), 5)

            print( non_zeros, zeros, acc)


class DomainAdaptationTrainer:

    def __init__(self, model, criterion, criterion_t, optimizer, optimizer_kl, scheduler, max_epochs, ae_model=None,
                 epochs_no_improve=8):
        self.model = model
        self.criterion = criterion
        self.criterion_t = criterion_t
        self.optimizer = optimizer
        self.optimizer_kl = optimizer_kl
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.ae_model = ae_model
        self.epochs_no_improve = epochs_no_improve
        self.model.to(device)
        self.kl_sim = []
        self.v_accs = []
        self.t_accs = []
        self.src_domain = ""
        self.tgt_domain = ""

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
            # src_iter = iter(training_loader)
            tgt_iter = iter(target_generator)

            # ###################### KL Similarity Optimalization ############################

            src_generator = DataLoader(training_loader.dataset, shuffle=True, batch_size=len(training_loader.dataset))
            tgt_generator = DataLoader(target_generator.dataset, shuffle=True, batch_size=len(target_generator.dataset))

            print('\n')
            for _ in range(1):
                # self.optimizer_kl.zero_grad()

                # with torch.no_grad():
                self.optimizer_kl.zero_grad()

                for idx, inputs, labels, domain_gt in src_generator:
                    inputs = inputs.to(device, torch.float)
                    src_encoded = self.ae_model(inputs)

                for idx, inputs, labels, domain_gt in tgt_generator:
                    inputs = inputs.to(device, torch.float)
                    tgt_encoded = self.ae_model(inputs)

                loss = KLDivergenceLoss()(src_encoded, tgt_encoded)
                print(loss.item())
            #
            # if loss.item() > 1.0:
            # loss.backward()
            # self.optimizer_kl.step()
            #
            self.optimizer_kl.zero_grad()
            # #################################################################################

            for idx, input, labels, src, loss_upd in batches:
                n_batch += 1

                # try:
                #     idx_tgt, input_tgt, labels_tgt, src_tgt = next(tgt_iter)
                # except:
                #     # tgt_iter = iter(target_generator)
                #     idx_tgt, input_tgt, labels_tgt, src_tgt = next(tgt_iter)

                if type(labels) == list:
                    labels = torch.stack(labels, dim=1)
                # if type(labels_tgt) == list:
                #     labels_tgt = torch.stack(labels_tgt, dim=1)

                # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices
                # max(1) will return the maximal value (and index in PyTorch) in this particular dimension.
                input, labels_, src = input.to(device, torch.float), torch.max(labels, 1)[1].to(device,
                                                                                                torch.long), src.to(
                    device, torch.float)
                # input_tgt, labels_tgt, src_tgt = input_tgt.to(device, torch.float), torch.max(labels_tgt, 1)[1].to(device, torch.long), src_tgt.to(device, torch.float)
                # if self.ae_model is not None:
                #     input = self.ae_model(input)
                # input = torch.cat([input, ae_output], 1)

                self.optimizer.zero_grad()

                # AutoEncoder domain discrepancy
                # _, src_batch_data, _, _ = next(src_iter)

                self.ae_model.zero_grad()

                # enc_src_out = self.ae_model(input)
                # enc_tgt_out = self.ae_model(input_tgt)
                #################################

                f1, f2, ft, _ = self.model(input)
                _loss = self.criterion(f1, f2, self.model.f1_1.weight, self.model.f2_1.weight,
                                       labels_)  # enc_src_out, enc_tgt_out, labels_)

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
                _tgt_acc = self.valid(target_generator, valid=False)

            self.scheduler.step(_valid_acc)

            # print('Train Len: ' + str(len(training_loader.dataset)))
            # print('Target Len: ' + str(len(target_generator.dataset)))

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

        self.kl_sim.append(round(loss.item(), 8))
        # self.kl_sim.append(0)
        self.v_accs.append(round(_valid_acc, 4))
        self.t_accs.append(round(_tgt_acc, 4))

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
        n_wrongs = []
        n_alls = []

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
                if f1_idx[_i] == f2_idx[_i] and f1_max[_i] > 0.95 and f2_max[_i] > 0.95:
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

            n_wrongs.append(n_all_wrong_labeled)
            n_alls.append(n_all)

            print(
                'Iteration: {}, wrong labeled count: {} on {} qualified, all {}. All wrong labeled: {} / {} [{}%]'.format(
                    _iter, n_wrong_labeled, n_all_qualified, len(idx), n_all_wrong_labeled, n_all,
                    round((n_all_wrong_labeled / n_all), 4) * 100))

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

        with open('tmp/metrics.txt', 'a+') as f:
            f.write(self.src_domain + ' ---> ' + self.tgt_domain + '\n')
            f.write('[' + ', '.join(map(str, self.kl_sim)) + ']' + '\n')
            f.write('[' + ', '.join(map(str, self.v_accs)) + ']' + '\n')
            f.write('[' + ', '.join(map(str, self.t_accs)) + ']' + '\n')
            f.write('[' + ', '.join(map(str, n_wrongs)) + ']' + '\n')
            f.write('[' + ', '.join(map(str, n_alls)) + ']' + '\n')
            f.write('\n')

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


def measure(predicted, ground_truth):
    return np.count_nonzero(ground_truth), len(ground_truth) - np.count_nonzero(ground_truth), np.round((ground_truth == predicted).sum() / len(ground_truth), 5)
