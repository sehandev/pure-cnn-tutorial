from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import split_dataset


def load_dataset(data_dir: Path):
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_dataset, valid_dataset = split_dataset(train_dataset, ratio=0.8)
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_dataset, valid_dataset, test_dataset


def create_dataloaders(
    data_dir: Path,
    train_batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
) -> (DataLoader, DataLoader, DataLoader):
    train_dataset, valid_dataset, test_dataset = load_dataset(data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_dataloader, valid_dataloader, test_dataloader
