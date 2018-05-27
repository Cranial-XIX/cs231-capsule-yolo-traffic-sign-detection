import torch
import torch.nn.functional as F

def cnn_loss(scores, y, params):
    return (-F.log_softmax(scores, dim=1).gather(1, y.unsqueeze(1))).sum()


def capsule_loss(scores, y, params):
    left = F.relu(0.9 - scores) ** 2
    right = F.relu(scores - 0.1) ** 2
    labels = torch.eye(param.n_classes).to(
        param.device).index_select(dim=0, index=y)
    margin_loss = labels * left + 0.5 * (1. - labels) * right
    return margin_loss.sum() / y.size(0)


def dark_d_loss():
    pass


def dark_r_loss():
    pass


def darkcapsule_loss():
    pass