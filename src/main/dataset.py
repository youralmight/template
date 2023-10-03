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

train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set)
valid_loader= DataLoader(valid_set)