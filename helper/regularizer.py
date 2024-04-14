#!./env python

import torch
import copy


class SWA:
    """
        Stochastic Weight Averaing
    """
    def __init__(self, net, trainloader, swa_start, swa_interval, name=''):

        self.net = net
        self.trainloader = trainloader
        self.name = name

        # swa setup
        self.swa_start = swa_start
        self.swa_interval = swa_interval
        print('[SWA%s] SWA start at epoch %i with interval %i' % (self.name, self.swa_start, self.swa_interval))
        self.swa_count = 0 # number of averaged models === number of epochs since swa started
        self.swa_net = None # copy the current model at the first step

    def update(self, epoch):
        if not (epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_interval == 0):
            return

        # swa updated
        self.__swa_update()
        self.swa_net.eval()

    def close(self):
        pass

    def __swa_update(self):
        if self.swa_net is None:
            print('[SWA%s] copy the current model at the first SWA step' % self.name)
            self.swa_net = copy.deepcopy(self.net)

        print('[SWA%s] # SWA update steps: %i' % (self.name, self.swa_count))
        alpha = 1.0 / (self.swa_count + 1)

        # Init swa model will be replaced by the model at the first update step, since the ratio is 1
        self._moving_average(self.swa_net, self.net, alpha)
        self.swa_count += 1
        self._bn_update(self.trainloader, self.swa_net)

    def _moving_average(self, net1, net2, alpha=1):
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

    def _bn_update(self, loader, model):
        """
            BatchNorm buffers update (if any).
            Performs 1 epochs to estimate buffers average using train dataset.

            :param loader: train dataset loader for buffers average estimation.
            :param model: model being update
            :return: None
        """
        if not self._check_bn(model):
            return
        model.train()
        momenta = {}
        model.apply(self._reset_bn)
        model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0
        for input, _, _ in loader:
            input = input.cuda()
            b = input.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input)
            n += b

        model.apply(lambda module: self._set_momenta(module, momenta))

    def _check_bn(self, model):

        def __check_bn(module, flag):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                flag[0] = True

        flag = [False]
        model.apply(lambda module: __check_bn(module, flag))
        return flag[0]
    
    def _reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
    
    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]


