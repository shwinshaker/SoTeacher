
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

from .loss import mse_soft

class ConsistencyRegularizer:
    def __init__(self, alpha, scheduler=None, start_at=None, epochs=None, trainsize=50000, n_classes=10):
        self.alpha = alpha
        self.scheduler = scheduler
        if scheduler is not None:
            self.weight_schedule = self.__get_weight_schedule(schedule=scheduler,
                                                              max_weight=alpha,
                                                              epochs=epochs,
                                                              start_at=start_at)
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.temperature = 1.0
        self.crl_criterion = mse_soft(temperature=self.temperature, num_classes=n_classes)

        self.crl_soft_labels = torch.zeros(trainsize, n_classes).cuda()
        self.epoch = 0

    def step(self, epoch):
        if self.scheduler is not None:
            self.alpha = self.weight_schedule(epoch)
            print('crl weight: %.3f' % self.alpha)

        self.epoch = epoch + 1

    def _criterion(self, outputs, targets, indices):

        crl_loss = torch.tensor([0.])
        if self.epoch > 0:
            soft_labels = self.crl_soft_labels[indices]
            crl_loss = self.crl_criterion(outputs, soft_labels)
            crl_loss = self.alpha * crl_loss

        # update soft label
        probas = F.softmax(outputs.detach() / self.temperature, dim=1)
        tau = 1. / (1. + self.epoch)
        self.crl_soft_labels[indices] *= 1. - tau
        self.crl_soft_labels[indices] += tau * probas

        return crl_loss.mean()

    def __get_weight_schedule(self, schedule, max_weight, epochs, start_at=100):
        if schedule == 'linear':
            def __schedule(e):
                return max_weight * (e + 1) / epochs
        elif schedule == 'step':
            # start at: the epoch when weight becomes nonzero
            def __schedule(e):
                if e < start_at:
                    return 0
                else:
                    return max_weight
        elif schedule == 'cycle':
            def __schedule(e):
                x_norm = 1 - (e + 1) / epochs
                return max_weight * np.sqrt(1 - x_norm**2)
        elif schedule == 'cosine':
            def __schedule(e):
                x_norm = 1 - (e + 1) / epochs
                return max_weight * np.cos(x_norm * np.pi / 2)
        else:
            raise NotImplementedError(schedule)
        return __schedule