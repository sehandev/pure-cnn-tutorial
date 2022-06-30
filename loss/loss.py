from torch import nn


def get_loss_fn():
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn
