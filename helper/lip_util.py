
import torch

def lip1Loss(net):
    loss = torch.tensor(0.).cuda()
    for name, module in net.named_modules():
        if not list(module.children()): # leaf
            if 'conv' in module._get_name().lower():
                assert(len(module.weight.size()) == 4)
                loss += torch.norm(module.weight, 1)
            elif 'batchnorm' in module._get_name().lower():
                loss += torch.max(torch.abs(module.weight) / torch.sqrt(module.running_var + module.eps))
            elif 'linear' in module._get_name().lower():
                loss += module.weight.abs().sum(dim=0).max()
            else:
                pass
    return loss
