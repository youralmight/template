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


class PLMnist(pl.LightningDataModule):
    def __init__(self, fake_arg, data_dir: str = "data/MNIST"):
        super().__init__()
        self.data_dir = data_dir
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    # called on main process before other multi-node processes
    # download, i.e. download data only once on the disk from a single process
    # tokenize. Since it’s a one time process, it is not recommended to do it on all processes
    # etc…
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    # this is performed on every GPU
    # count number of classes
    # build vocabulary
    # perform train/val/test splits
    # create datasets
    # apply transforms (defined explicitly in your datamodule)
    # etc…
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


###################################################### independant train, validation, test set
# transform = transforms.ToTensor()
# train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
# test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# train_loader = DataLoader(train_set)
# test_loader = DataLoader(test_set)

###################################################### splited train, validation, test set
