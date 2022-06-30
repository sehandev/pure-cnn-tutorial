import pathlib
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def get_device(device: str) -> torch.device:
    if device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    return device


def check_dir(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        if not path.is_dir():
            raise Exception(f"path must be dir: {path}")
        return path

    path.mkdir(parents=True)
    print(f"mkdir: {path}")
    return path


def split_dataset(
    dataset: Dataset,
    ratio: float = None,
    size: int = None,
) -> (Dataset, Dataset):
    if ratio is None and size is None:
        raise Exception("ratio or size must be passed")
    if ratio is not None and size is not None:
        raise Exception("ratio and size should not passed at the same time")

    full_length = len(dataset)
    if ratio is not None:
        if not (0.0 < ratio < 1.0):
            raise Exception(f"Dataset length: {full_length} / Ratio: {ratio} ")
        size_1 = int(full_length * ratio)
    if size is not None:
        if not (0 < size < full_length):
            raise Exception(f"Dataset length: {full_length} / Size: {size} ")
        size_1 = size
    size_2 = full_length - size_1
    dataset_1, dataset_2 = random_split(dataset, [size_1, size_2])
    return dataset_1, dataset_2
