from __future__ import print_function

import os
import argparse
import json
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

from helper.util import get_lr, Dict2Obj
from helper.logger import Logger
from helper.crl_util import Crl
from helper.consist_util import ConsistencyRegularizer
from helper.regularizer import SWA
from helper.lip_util import lip1Loss


def main(config_file):
    # load config
    opt = {}
    with open(config_file, 'rt') as f:
        opt.update(json.load(f))
    opt = Dict2Obj(opt)


    best_acc = 0
    best_loss = np.inf

    # environment set
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # -- SWA setup
    if opt.swa:
        swaer = SWA(model, 
                    trainloader=train_loader,
                    swa_start=opt.swa_start, 
                    swa_interval=1,
                    name='test')
    _is_swa = lambda epoch: opt.swa and epoch >= opt.swa_start

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # loss
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    train_criterion = lambda inputs, outputs, targets, indices, model: criterion(outputs, targets) # dummy
    if opt.crl:
        print('-------- confidence-aware learning!')
        crler = Crl(opt.crl_alpha, scheduler=opt.crl_scheduler, start_at=opt.crl_start_at, epochs=opt.epochs,
                    trainsize=train_loader.size, n_classes=train_loader.n_classes)
        train_criterion = lambda inputs, outputs, targets, indices, model: criterion(outputs, targets) + crler._criterion(outputs, targets, indices)

    if opt.lip:
        print('-------- Lipschitz regularization!')
        train_criterion = lambda inputs, outputs, targets, indices, model: criterion(outputs, targets) + opt.lip_alpha * lip1Loss(model) 

    if opt.consist:
        print('-------- Consistency regularization!')
        consister = ConsistencyRegularizer(opt.consist_alpha, scheduler=opt.consist_scheduler, start_at=opt.consist_start_at, epochs=opt.epochs,
                      trainsize=train_loader.size, n_classes=train_loader.n_classes)
        train_criterion = lambda inputs, outputs, targets, indices, model: criterion(outputs, targets) + consister._criterion(outputs, targets, indices)
        if opt.lip:  # legacy issue
            train_criterion = lambda inputs, outputs, targets, indices, model: criterion(outputs, targets) \
                                                                               + consister._criterion(outputs, targets, indices) \
                                                                               + opt.lip_alpha * lip1Loss(model) 

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # txt logger
    logger_txt = Logger(os.path.join(opt.save_folder, 'log.txt'), title='log')
    logger_txt.set_names(['Epoch', 'lr', 'Time-elapse(Min)',
                          'Train-Loss', 'Test-Loss', 
                          'Train-Acc', 'Test-Acc', 'Test-Acc-Top5'])

    # routine
    time_start = time.time()
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, train_criterion, optimizer, opt)
        if opt.crl:
            crler.step(epoch)
        if opt.consist:
            consister.step(epoch)
        time2 = time.time()
        print('epoch {}, total training time {:.2f}'.format(epoch, time2 - time1))

        time1 = time.time()
        if _is_swa(epoch):
            swaer.update(epoch)
            test_acc, test_acc_top5, test_loss = validate(val_loader, swaer.swa_net, criterion, opt)
        else:
            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        time2 = time.time()
        print('epoch {}, total test time {:.2f}'.format(epoch, time2 - time1))

        # tb logger
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # txt logger
        logs = [epoch, get_lr(optimizer), (time.time() - time_start)/60]
        logs += [train_loss, test_loss,
                 train_acc, test_acc, test_acc_top5]
        logger_txt.append(logs)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            print('Best acc %.3f at epoch %i' % (best_acc, epoch))
            state = {
                'epoch': epoch,
                'model': swaer.swa_net.state_dict() if _is_swa(epoch) else model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            torch.save(state, save_file)

        # save the best model with loss
        if test_loss < best_loss:
            best_loss = test_loss
            print('Best loss %.3f at epoch %i' % (best_loss, epoch))
            state = {
                'epoch': epoch,
                'model': swaer.swa_net.state_dict() if _is_swa(epoch) else model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best_loss.pth'.format(opt.model))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('Best accuracy: %.3f' % best_acc)
    print('Final accuracy: %.3f' % test_acc)

    # save model
    if _is_swa(epoch):
        print('Save SWA model.')
    state = {
        'epoch': epoch,
        'opt': opt,
        'model': swaer.swa_net.state_dict() if _is_swa(epoch) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':

    # get config file
    parser = argparse.ArgumentParser('config file')
    parser.add_argument('--config_file', '-c', type=str, help='config file')
    config = parser.parse_args()
    print('Config file: ', config.config_file, end='\n\n')

    main(config.config_file)
