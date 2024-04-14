

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_soft(reduction='mean', num_classes=10, soft_label=True, temperature=1.0):
    def mse(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        probas = torch.nn.functional.softmax(outputs / temperature, dim=1)
        if not soft_label:
            if len(labels.size()) == 1:
                # provided labels are hard label
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            else:
                # provided labels are soft label
                _, labels = torch.max(labels, 1)
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        if reduction == 'mean':
            return nn.MSELoss(reduction=reduction)(probas, labels) * num_classes # mse_loss in pytorch average over dimension too
        return nn.MSELoss(reduction='none')(probas, labels).sum(dim=1) # reduce class dimension, keep batch dimension
    return mse


def ce_soft(reduction='mean', num_classes=10, soft_label=True, temperature=1.0):
    def ce(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        if not soft_label:
            if len(labels.size()) > 1:
                _, labels = torch.max(labels, 1)
            return torch.nn.functional.cross_entropy(outputs, labels, reduction=reduction)
        
        assert(len(labels.size()) == 2), ('soft labels required, got size: ', labels.size())
        log_probas = torch.nn.functional.log_softmax(outputs / temperature, dim=1)
        nll = -(log_probas * labels) * temperature**2 # normalize
        loss = nll.sum(dim=1)
        if reduction == 'mean':
            return loss.mean()
        return loss
    return ce


def ranking_soft(reduction='mean', temperature=1.0, gamma=0.0):
    ranking_criterion = nn.MarginRankingLoss(margin=0.0, reduction='none')
    def ranking(outputs, soft_labels):
        # make input pair
        softmax = F.softmax(outputs / temperature, dim=1)
        rank_input1 = softmax
        rank_input2 = rank_input1[torch.roll(torch.arange(len(rank_input1)), -1)]
        rank_label1 = soft_labels
        rank_label2 = rank_label1[torch.roll(torch.arange(len(rank_label1)), -1)]

        # calc target, margin
        rank_target, rank_margin, rank_weight = get_target_margin(rank_label1, rank_label2, gamma=gamma)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        loss = ranking_criterion(rank_input1.flatten(),
                                 rank_input2.flatten(),
                                 rank_target.flatten())
        loss = loss.reshape(rank_input1.size())
        loss = loss * rank_weight
        loss = loss.sum(dim=1)
        if reduction == 'mean':
            loss = loss.mean()
        return loss
    return ranking

def get_target_margin(target1, target2, gamma=0):
    # calc target
    greater = (target1 > target2).float()
    less = (target1 < target2).float() * (-1)
    target = greater + less

    # calc margin
    margin = torch.abs(target1 - target2)

    # calc weights
    weights = torch.tensor(1.0).to(target1.device)
    if gamma > 0:
        # weights = (1 - (target1 + target2) / 2.0)**gamma
        weights = (target1.max(1)[0] + target2.max(1)[0]) / 2.0

        # p = (target1 + target2) / 2.0
        # eps = 1e-8
        # weights = torch.sum(-p * (p + eps).log(), dim=1)
        
        weights = weights.repeat(target1.size(1), 1).T
        weights = weights**gamma
        weights = weights / weights.sum() * weights.nelement()
        
    return target, margin, weights

