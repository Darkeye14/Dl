import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _worker_init_fn(worker_id):
    # Ensure each worker has a different, deterministic seed
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_dataloaders(
    dataset: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    root: str = "data",
) -> Tuple[DataLoader, DataLoader]:
    assert dataset in {"mnist", "cifar10"}

    if dataset == "mnist":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        train_ds = datasets.MNIST(root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(root, train=False, download=True, transform=test_tf)
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader
