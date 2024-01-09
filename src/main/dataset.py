import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms

###################################################### independant train, validation, test set
# transform = transforms.ToTensor()
# train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
# test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# train_loader = DataLoader(train_set)
# test_loader = DataLoader(test_set)

###################################################### splited train, validation, test set

