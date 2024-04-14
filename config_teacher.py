import socket
import argparse
import json
import os

from helper.util import check_path

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--swa', action='store_true', help='swa (default: off)')
    parser.add_argument('--swa_start', type=int, default=150, help='epoch when swa starts')

    # regularization
    parser.add_argument('--crl', action='store_true', help='crl (default: off)')
    parser.add_argument('--crl_alpha', type=float, default=1, help='crl alpha')
    parser.add_argument('--crl_scheduler', type=str, default=None, help='crl weight scheduler')
    parser.add_argument('--crl_start_at', type=int, default=50, help='crl weight start at for step scheduler')

    parser.add_argument('--lip', action='store_true', help='lip (default: off)')
    parser.add_argument('--lip_alpha', type=float, default=0.00001, help='lip alpha')

    parser.add_argument('--consist', action='store_true', help='consist (default: off)')
    parser.add_argument('--consist_alpha', type=float, default=1, help='consist alpha')
    parser.add_argument('--consist_scheduler', type=str, default=None, help='consist weight scheduler')
    parser.add_argument('--consist_start_at', type=int, default=50, help='consist weight start at for step scheduler')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay)
    if opt.lip:
        opt.model_name += '_lip_alpha=%g' % opt.lip_alpha
    if opt.crl:
        opt.model_name += '_crl_alpha=%g' % opt.crl_alpha
        if opt.crl_scheduler is not None:
            opt.model_name += '_%s' % opt.crl_scheduler
            if opt.crl_scheduler == 'step':
                opt.model_name += '_start_at_%i' % opt.crl_start_at
    if opt.consist:
        opt.model_name += '_consist_alpha=%g' % opt.consist_alpha
        if opt.consist_scheduler is not None:
            opt.model_name += '_%s' % opt.consist_scheduler
            if opt.consist_scheduler == 'step':
                opt.model_name += '_start_at_%i' % opt.consist_start_at
    if opt.swa:
        opt.model_name += '_swa=%g' % opt.swa_start
    opt.model_name += '_trial_%i' % opt.trial
    print(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    check_path(opt.tb_folder)
    print('tb folder: %s' % opt.tb_folder, end='\n\n')

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    check_path(opt.save_folder)
    print('save folder: %s' % opt.save_folder, end='\n\n')

    return opt

if __name__ == '__main__':
    opt = parse_option()

    # Save setting to json
    with open('tmp/config.tmp', 'wt') as f:
        json.dump(vars(opt), f, indent=4)