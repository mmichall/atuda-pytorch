import argparse
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss, MSELoss, _Loss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast
import torch
from torch.utils.data import DataLoader

from data_set import as_one_dataloader, load_data
from nn.ae_gan import AE_Generator
from nn.loss import MultiViewLoss, ReversalLoss, MSEWithDivergenceLoss
from nn.model import SimpleAutoEncoder, ATTFeedforward, ModelWithTemperature
from data_set import train_valid_target_split
from nn.trainer import AutoEncoderTrainer, DomainAdaptationTrainer, AEGeneratorTrainer
from stats import get_unique_per_set_words


def run(args):
    domains_summary()
    parameters_summary()

    train_params = {'batch_size': args.train_batch_size, 'shuffle': args.train_data_set_shuffle}

    if args.model == 'AutoEncoder':
        ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        ae_model.summary()

        optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience)
        #criterion = args.loss
        criterion = MSEWithDivergenceLoss()
        #criterion = MSELoss()

        # src_domain_data_set, tgt_domain_data_set = load_data('kitchen', 'books', verbose=True)
        # words_to_reconstruct = get_unique_per_set_words(src_domain_data_set, tgt_domain_data_set)

        #data_generator = as_one_dataloader(args.src_domain, args.tgt_domain, train_params, denoising_factor=args.denoising_factor, return_input=True)#,  words_to_reconstruct=words_to_reconstruct)
        src_domain_data_set, tgt_domain_data_set = load_data('books', 'kitchen', verbose=True, return_input=True)
        src_domain_data_set.denoising_factor = args.denoising_factor
        tgt_domain_data_set.denoising_factor = args.denoising_factor

        src_generator = DataLoader(src_domain_data_set, batch_size=args.train_batch_size,
                                   shuffle=args.train_data_set_shuffle)
        tgt_generator = DataLoader(tgt_domain_data_set, batch_size=args.train_batch_size,
                                   shuffle=args.train_data_set_shuffle)

        trainer = AutoEncoderTrainer(ae_model, criterion, optimizer, scheduler, args.max_epochs, epochs_no_improve=args.epochs_no_improve)
        trainer.fit(src_generator, tgt_generator)
        torch.save(ae_model.state_dict(), args.ae_model_file)
        print('Model was saved in {} file.'.format(args.ae_model_file))

    elif args.model == 'AE_Generator':
        ae_model = AE_Generator(5000, 250)
        ae_model.summary()

        g_optimizer = torch.optim.Adam(ae_model.encoder.parameters(), lr=args.learning_rate)
        d_optimizer = torch.optim.Adam(ae_model.domain_discriminator.parameters(), lr=0.001)
        ae_optimizer = torch.optim.Adam(list(ae_model.encoder.parameters()) + list(ae_model.decoder.parameters()), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(g_optimizer, mode='min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience)

        reconstruction_criterion = BCEWithLogitsLoss()
        discrimination_criterion = BCEWithLogitsLoss()

        src_domain_data_set, tgt_domain_data_set = load_data('books', 'kitchen', verbose=True, return_input=True)
        src_domain_data_set.denoising_factor = args.denoising_factor
        tgt_domain_data_set.denoising_factor = args.denoising_factor

        src_generator = DataLoader(src_domain_data_set, batch_size=args.train_batch_size, shuffle=args.train_data_set_shuffle)
        tgt_generator = DataLoader(tgt_domain_data_set, batch_size=args.train_batch_size, shuffle=args.train_data_set_shuffle)


        trainer = AEGeneratorTrainer(ae_model, reconstruction_criterion, discrimination_criterion, discrimination_criterion, g_optimizer, d_optimizer, ae_optimizer, scheduler, args.max_epochs,
                                     epochs_no_improve=args.epochs_no_improve)

        trainer.fit(src_generator, tgt_generator)
        torch.save(ae_model.state_dict(), args.ae_model_file)
        print('Model was saved in {} file.'.format(args.ae_model_file))

    elif args.model == 'ATTFeedforward':
        src_domain_data_set, tgt_domain_data_set = load_data(args.src_domain, args.tgt_domain, verbose=True)
        train_generator, valid_generator, target_generator = train_valid_target_split(src_domain_data_set, tgt_domain_data_set, train_params)

        ae_model = None
        if args.auto_encoder_embedding is not None:
            # ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
            ae_model = SimpleAutoEncoder(ast.literal_eval(args.autoencoder_shape))
            ae_model.load_state_dict(torch.load(args.auto_encoder_embedding))
            ae_model.set_train_mode(False)
            # ae_model.froze()

        attff_model = ATTFeedforward(args.attff_input_size, args.attff_hidden_size, ae_model)
        attff_model.summary()

        criterion = MultiViewLoss()
        criterion_t = CrossEntropyLoss()

        optimizer = torch.optim.Adagrad(attff_model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.reduce_lr_factor, patience=10)

        trainer = DomainAdaptationTrainer(attff_model, criterion, criterion_t, optimizer, scheduler, args.max_epochs,
                                          ae_model=ae_model, epochs_no_improve=args.epochs_no_improve)

        if args.load_attnn_model:
            attff_model.load_state_dict(torch.load(args.attnn_model_file.format(args.attff_input_size, args.attff_hidden_size)))
        else:
            trainer.fit(train_generator, valid_generator, target_generator=target_generator, max_epochs=args.max_epochs)

            model_file = args.attnn_model_file.format(args.attff_input_size, args.attff_hidden_size)
            torch.save(attff_model.state_dict(), model_file)
            print('Model was saved in {} file.'.format(model_file))

        trainer.pseudo_label(train_generator, valid_generator, tgt_domain_data_set, iterations=args.pseudo_label_iterations, train_params=train_params, max_epochs=args.max_epochs)

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

    # Experiment parameters
    parser.add_argument('--data_set', required=False, default='amazon')
    parser.add_argument('--src_domain', required=False, help='the source domain.', default='books')
    parser.add_argument('--tgt_domain', required=False, help='the target domain.', default='kitchen')

    # Training parameters
    parser.add_argument('--model', required=False, default='AutoEncoder')
    parser.add_argument('--max_epochs', required=False, type=int, default=500)
    parser.add_argument('--train_batch_size', required=False, type=int, default=8)
    parser.add_argument('--train_data_set_shuffle', required=False, type=bool, default=True)
    parser.add_argument('--learning_rate', required=False, type=float, default=1.0e-04)
    parser.add_argument('--reduce_lr_factor', required=False, type=float, default=0.5)
    parser.add_argument('--reduce_lr_patience', required=False, type=int, default=3)
    parser.add_argument('--denoising_factor', required=False, type=float, default=0.5)
    parser.add_argument('--epochs_no_improve', required=False, type=float, default=3)
    parser.add_argument('--loss', required=False, type=_Loss, default=MSELoss(reduction='mean'))
    parser.add_argument('--auto_encoder_embedding', required=False, default='tmp/auto_encoder_5000_500_250_mse_kl_500iter_2.pt')
    parser.add_argument('--load_attnn_model', required=False, type=bool, default=False)
    parser.add_argument('--pseudo_label_iterations', required=False, type=int, default=10)

    # Models parameters
    parser.add_argument('--autoencoder_shape', required=False, default='(5000, 500, 250)')
    parser.add_argument('--attff_input_size', required=False, type=int, default=5000)
    parser.add_argument('--attff_hidden_size', required=False, type=int, default=50)
    parser.add_argument('--ae_model_file', required=False, default='tmp/auto_encoder_5000_500_250_mse_kl_500iter_2.pt')
    parser.add_argument('--attnn_model_file', required=False, default='tmp/attnn_model_{}_{}.pt')

    args = parser.parse_args()
    run(args)
