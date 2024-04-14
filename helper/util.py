from __future__ import print_function

import torch
import numpy as np


import os
import shutil
import sys

class Dict2Obj:
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])


def get_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    assert(len(lrs) == 1)
    return lrs[0]


import re
def get_teacher_name(model_path):
    """parse teacher name"""
    return re.findall(r'models\/(\w+)_cifar.*\/.*$', model_path)[0]

    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        option = input('Path %s already exists. Delete[d], Terminate[*]? ' % path)
        if option.lower() == 'd':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('Terminated.')
            sys.exit(2)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass
