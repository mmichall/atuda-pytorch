import argparse
from torch.nn.modules.loss import BCELoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast
import torch
from data_set import as_one_dataloader
from nn.model import StackedAutoEncoder
from data_set import train_valid_target_split
from nn.trainer import AutoEncoderTrainer


def run(args):
    print('> SOURCE domain: {}'.format(args.src_domain))
    print('> TARGET domain: {}'.format(args.tgt_domain))

    train_params = {'batch_size': args.train_batch_size, 'shuffle': args.train_data_set_shuffle}

    if args.model == 'AutoEncoder':
        ae_model = StackedAutoEncoder(ast.literal_eval(args.autoencoder_shape))
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3)
        criterion = MSELoss()
        data_generator = as_one_dataloader(args.src_domain, args.tgt_domain, train_params, denoising_factor=0.5)
        trainer = AutoEncoderTrainer(ae_model, criterion, optimizer, scheduler, args.max_epochs, epochs_no_improve=3)
        trainer.fit(data_generator)
        torch.save(ae_model.state_dict(), 'tmp/dae_model_500.pt')
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

    # Models parameters
    parser.add_argument('--autoencoder_shape', required=False, default='(5000,1000,250)')

    args = parser.parse_args()
    run(args)
