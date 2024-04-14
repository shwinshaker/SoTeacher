
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Crl:
    """
        classwise-uncertainty regularization learning
    """
    def __init__(self, alpha, scheduler=None, start_at=None, epochs=None,
                 trainsize=50000, n_classes=10):

        self.alpha = alpha
        self.scheduler = scheduler
        if scheduler is not None:
            self.weight_schedule = self.__get_weight_schedule(schedule=scheduler,
                                                              max_weight=alpha,
                                                              epochs=epochs,
                                                              start_at=start_at)
        self.crl_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

        self.correctness_history = History(trainsize)
        self.rank_target = 'softmax'
        self.dataset = 'cifar100'
        self.n_classes = n_classes


    def step(self, epoch):
        if self.scheduler is not None:
            self.alpha = self.weight_schedule(epoch)
            print('crl weight: %.3f' % self.alpha)

        self.correctness_history.max_correctness_update(epoch)


    def _criterion(self, outputs, targets, indices):

       # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if self.rank_target == 'softmax':
            conf = F.softmax(outputs, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif self.rank_target == 'entropy':
            if self.dataset == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = negative_entropy(outputs,
                                          normalize=True,
                                          max_value=value_for_normalizing)
        # margin
        elif self.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(outputs), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]

        # make input pair
        idx = indices.cuda()
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = self.correctness_history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        ranking_loss = self.crl_criterion(rank_input1,
                                          rank_input2,
                                          rank_target)

        # total loss
        ranking_loss = self.alpha * ranking_loss

        _, preds = outputs.max(1)
        corrects = preds.eq(targets)
        self.correctness_history.correctness_update(idx, corrects, outputs)


        return ranking_loss.mean()

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



# rank target entropy
def negative_entropy(data, normalize=False, max_value=None):
    softmax = F.softmax(data, dim=1)
    log_softmax = F.log_softmax(data, dim=1)
    entropy = softmax * log_softmax
    entropy = -1.0 * entropy.sum(dim=1)
    # normalize [0 ~ 1]
    if normalize:
        normalized_entropy = entropy / max_value
        return -normalized_entropy

    return -entropy


# correctness history class
class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness, output):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        data_idx2 = data_idx2.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().cuda()
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin