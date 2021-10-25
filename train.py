import argparse
import os
import numpy as np

from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, _Loss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast
import torch

from data_set import load_data
from nn.loss import MultiViewLoss
from nn.model import SimpleAutoEncoder, ATTFeedforward
from data_set import train_valid_target_split
from nn.trainer import AutoEncoderTrainer, DomainAdaptationTrainer


def run(args):
    domains_summary()
    parameters_summary()

    train_params = {'batch_size': args.train_batch_size, 'shuffle': args.train_data_set_shuffle}

    if args.model == 'AutoEncoder':
        ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        print(args.autoencoder_shape)
        ae_model.summary()

        optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.learning_rate)
        optimizer_kl = torch.optim.Adam(ae_model.encoder.parameters(), lr=args.learning_rate_kl)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_lr_factor,
                                      patience=args.reduce_lr_patience)
        criterion = BCEWithLogitsLoss()

        src_domain_data_set, tgt_domain_data_set = load_data(args.src_domain, args.tgt_domain, verbose=True, return_input=True)

        trainer = AutoEncoderTrainer(args.src_domain, args.tgt_domain, ae_model, criterion, optimizer, optimizer_kl,
                                     scheduler, args.max_epochs, epochs_no_improve=args.epochs_no_improve,
                                     model_file=args.ae_model_file, kl_threshold=float(args.kl_threshold))
        trainer.fit(src_domain_data_set, tgt_domain_data_set, denoising_factor=args.denoising_factor,
                    batch_size=args.train_batch_size)

    elif args.model == 'ATTFeedforward':
        src_domain_data_set, tgt_domain_data_set = load_data(args.src_domain, args.tgt_domain, verbose=True)
        train_generator, valid_generator, target_generator = train_valid_target_split(src_domain_data_set,
                                                                                      tgt_domain_data_set, train_params)

        ae_embedding_path = None
        ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        if args.auto_encoder_embedding is not None:
            ae_embedding_path = args.auto_encoder_embedding
            # ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
            ae_model.load_state_dict(torch.load(args.auto_encoder_embedding))
        else:
            ae_prefix = 'auto_encoder_' + args.src_domain + '_' + args.tgt_domain
            for filename in os.listdir('tmp'):
                if filename.startswith(ae_prefix):
                    ae_embedding_path = filename
                    print('AutoEncoder embedding ' + filename + ' loaded\n')
                    ae_model.load_state_dict(torch.load('tmp/' + filename))
                    break

        if not ae_embedding_path:
            raise RuntimeError('Failed to load AutoEncoder embedding')
        ae_model.froze()

        # attff_model = ATTFeedforward(args.attff_input_size, 250, None)
        attff_model = ATTFeedforward(args.attff_input_size, args.attff_hidden_size, ae_model)
        attff_model.summary()

        criterion = MultiViewLoss()
        criterion_t = CrossEntropyLoss()

        optimizer = torch.optim.Adagrad(attff_model.parameters(), lr=args.learning_rate)
        optimizer_kl = torch.optim.Adagrad(ae_model.encoder.parameters(), lr=args.learning_rate_kl)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.reduce_lr_factor, patience=4)

        trainer = DomainAdaptationTrainer(attff_model, criterion, criterion_t, optimizer, optimizer_kl, scheduler,
                                          args.max_epochs,
                                          ae_model=ae_model, epochs_no_improve=args.epochs_no_improve)
        trainer.src_domain = args.src_domain
        trainer.tgt_domain = args.tgt_domain

        if args.load_attnn_model:
            attff_model.load_state_dict(
                torch.load(args.attnn_model_file.format(args.attff_input_size, args.attff_hidden_size)))
        else:
            trainer.fit(train_generator, valid_generator, target_generator=target_generator, max_epochs=args.max_epochs)

            model_file = args.attnn_model_file.format(args.attff_input_size, args.attff_hidden_size)
            torch.save(attff_model.state_dict(), model_file)
            print('Model was saved in {} file.'.format(model_file))

        for _ in range(3):
            # ten sam autoencoder
            trainer.pseudo_label(train_generator, valid_generator, tgt_domain_data_set,
                                 iterations=args.pseudo_label_iterations, train_params=train_params,
                                 max_epochs=args.max_epochs)



def get_ae_model():
    ae_embedding_path = None
    ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
    if args.auto_encoder_embedding is not None:
        ae_embedding_path = args.auto_encoder_embedding
        ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        ae_model.load_state_dict(torch.load(args.auto_encoder_embedding))
    else:
        ae_prefix = 'auto_encoder_' + args.src_domain + '_' + args.tgt_domain
        for filename in os.listdir('tmp'):
            if filename.startswith(ae_prefix):
                ae_embedding_path = filename
                print('AutoEncoder embedding ' + filename + ' loaded\n')
                ae_model.load_state_dict(torch.load('tmp/' + filename))
                break

    if not ae_embedding_path:
        raise RuntimeError('Failed to load AutoEncoder embedding')
    ae_model.froze()
    return ae_model


def domains_summary():
    print('\n> Domains:')
    print('+\t SOURCE domain: {}'.format(args.src_domain))
    print('+\t TARGET domain: {}'.format(args.tgt_domain) + '\n')


def parameters_summary():
    print('> Parameters of training {} model:'.format(args.model))
    print('+\t data_loader parameters: {}'.format(
        {'batch_size': args.train_batch_size, 'shuffle': args.train_data_set_shuffle}))
    print('+\t loss: {}'.format(args.loss))
    print('+\t lr: {}'.format(args.learning_rate))
    print('+\t input denoising factor: {}'.format(args.denoising_factor))
    print('+\t scheduler parameters: {}'.format(
        {'lr factor': args.reduce_lr_factor, 'lr_patience': args.reduce_lr_patience}))
    print('+\t max epochs: {}'.format(args.max_epochs))
    print('+\t epochs no improve: {}'.format(args.epochs_no_improve))
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src_domain = "dvd"
    tgt_domain = "electronics"
    kl_threshold = 999 # only for file name ?

    # Experiment parameters
    parser.add_argument('--data_set', required=False, default='amazon')
    parser.add_argument('--src_domain', required=False, help='the source domain.', default=src_domain)
    parser.add_argument('--tgt_domain', required=False, help='the target domain.', default=tgt_domain)
    parser.add_argument('--kl_threshold', required=False, help='Kullback-Laibler Divergence threshold',
                        default=kl_threshold)

    # Training parameters
    parser.add_argument('--model', required=False, default='AutoEncoder')
    parser.add_argument('--max_epochs', required=False, type=int, default=400)  # 400
    parser.add_argument('--train_batch_size', required=False, type=int, default=8) # 8
    parser.add_argument('--train_data_set_shuffle', required=False, type=bool, default=True)
    parser.add_argument('--learning_rate', required=False, type=float, default=1.0e-02)  # for autoencoder learning: 1.0e-03
    parser.add_argument('--learning_rate_kl', required=False, type=float, default=1.0e-02)
    parser.add_argument('--reduce_lr_factor', required=False, type=float, default=0.5)
    parser.add_argument('--reduce_lr_patience', required=False, type=int, default=3)
    parser.add_argument('--denoising_factor', required=False, type=float, default=0.7)
    parser.add_argument('--epochs_no_improve', required=False, type=float, default=8)
    parser.add_argument('--loss', required=False, type=_Loss, default=MSELoss(reduction='mean'))
    parser.add_argument('--auto_encoder_embedding', required=False, default='tmp/auto_encoder_dvd_electronics__5000_5000_bce_0.00113_kl_1_epoch_156.pt')
    parser.add_argument('--load_attnn_model', required=False, type=bool, default=False)
    parser.add_argument('--pseudo_label_iterations', required=False, type=int, default=10)

    # Models parameters
    parser.add_argument('--autoencoder_shape', required=False, default='(5000, 3000)')
    parser.add_argument('--attff_input_size', required=False, type=int, default=5000)
    parser.add_argument('--attff_hidden_size', required=False, type=int, default=50)
    parser.add_argument('--ae_model_file', required=False, default='tmp/auto_encoder_{}_{}_' + '_5000_3000_bce_{}_BASELINE-encoder_{}.pt')
    parser.add_argument('--attnn_model_file', required=False, default='tmp/attnn_model_{}_{}.pt')

    args = parser.parse_args()
    run(args)
