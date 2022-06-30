from torch.optim import SGD


def get_optimizer(
    model,
    lr: float = 1e-2,
):
    optimizer = SGD(model.parameters(), lr=lr)
    return optimizer
