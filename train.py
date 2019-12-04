import argparse
from torch.nn.modules.loss import BCELoss, MSELoss, _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast
import torch
from data_set import as_one_dataloader
from nn.model import StackedAutoEncoder
from data_set import train_valid_target_split
from nn.trainer import AutoEncoderTrainer


def run(args):
    domains_summary()
    print('> Parameters of training {} model:'.format(args.model))

    train_params = {'batch_size': args.train_batch_size, 'shuffle': args.train_data_set_shuffle}
    print('+\t data_loader parameters: {}'.format(train_params))
    print('+\t loss: {}'.format(args.loss))
    print('+\t lr: {}'.format(args.learning_rate))
    print('+\t input denoising factor: {}'.format(args.denoising_factor))
    print('+\t scheduler parameters: {}'.format({'lr factor': args.reduce_lr_factor, 'lr_patience': args.reduce_lr_patience}))
    print('+\t max epochs: {}'.format(args.max_epochs))
    print('+\t epochs no improve: {}'.format(args.epochs_no_improve))
    print('\n')

    if args.model == 'AutoEncoder':
        ae_model = StackedAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        ae_model.summary()
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.reduce_lr_factor, patience=args.reduce_lr_patience)
        criterion = args.loss
        data_generator = as_one_dataloader(args.src_domain, args.tgt_domain, train_params, denoising_factor=args.denoising_factor)
        trainer = AutoEncoderTrainer(ae_model, criterion, optimizer, scheduler, args.max_epochs, epochs_no_improve=args.epochs_no_improve)
        trainer.fit(data_generator)
        torch.save(ae_model.state_dict(), args.model_file)
    else :
        train_generator, valid_generator, target_generator = train_valid_target_split(args.src_domain, args.tgt_domain,
                                                                                      train_params)
    #
    # model = ATTFeedforward(5250, 50)
    #
    # criterion = MultiViewLoss()
    # criterion_t = BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3)
    #
    # trainer = DomainAdaptationTrainer(ae_model, model, criterion, BCELoss(), optimizer, scheduler, args.max_epoch)
    # trainer.fit(train_generator, valid_generator, target_generator, dictionary)


def domains_summary():
    print('\n> Domains:')
    print('+\t SOURCE domain: {}'.format(args.src_domain))
    print('+\t TARGET domain: {}'.format(args.tgt_domain) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment parameters
    parser.add_argument('--data_set', required=False, default='amazon')
    parser.add_argument('--src_domain', required=False, help='the source domain.', default='books')
    parser.add_argument('--tgt_domain', required=False, help='the target domain.', default='kitchen')

    # Training parameters
    parser.add_argument('--model', required=False, default='AutoEncoder')
    parser.add_argument('--max_epochs', required=False, type=int, default=100)
    parser.add_argument('--train_batch_size', required=False, type=int, default=8)
    parser.add_argument('--train_data_set_shuffle', required=False, type=bool, default=True)
    parser.add_argument('--learning_rate', required=False, type=float, default=1.0e-04)
    parser.add_argument('--reduce_lr_factor', required=False, type=float, default=0.2)
    parser.add_argument('--reduce_lr_patience', required=False, type=int, default=3)
    parser.add_argument('--denoising_factor', required=False, type=float, default=0.5)
    parser.add_argument('--epochs_no_improve', required=False, type=float, default=3)
    parser.add_argument('--loss', required=False, type=_Loss, default=MSELoss(reduction='mean'))

    # Models parameters
    parser.add_argument('--autoencoder_shape', required=False, default='(5000,1000,250)')
    parser.add_argument('--model_file', required=False, default='model.pth')

    args = parser.parse_args()
    run(args)
