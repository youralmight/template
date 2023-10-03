import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

###################################################### independant train, validation, test set
# transform = transforms.ToTensor()
# train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
# test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# train_loader = DataLoader(train_set)
# test_loader = DataLoader(test_set)

###################################################### splited train, validation, test set

train_set = datasets.MNIST(
    root="data/MNIST", download=True, train=True, transform=transforms.ToTensor()
)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    train_set, [train_set_size, valid_set_size], generator=seed
)

# cpu core count workders
train_loader = DataLoader(
    train_set,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=os.cpu_count(),
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor=2,
    persistent_workers=False,
)

valid_loader = DataLoader(
    valid_set,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=os.cpu_count(),
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor=2,
    persistent_workers=False,
)
