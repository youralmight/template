
import pdb
import os
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from model import Encoder,Decoder,LitAutoEncoder
import traceback

############################### model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
autoencoder = (
    LitAutoEncoder.load_from_checkpoint("/path/to/checkpoint.ckpt", **{})
    if False
    else autoencoder
)

############################### data

train_set = MNIST(
    root="data/MNIST", download=True, train=True, transform=transforms.ToTensor()
)
# use 20% of training data for validation
valid_set_size = int(len(train_set) * 0.1)
test_set_size = int(len(train_set) * 0.1)
train_set_size = len(train_set) - valid_set_size - test_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set, test_set = torch.utils.data.random_split(
    train_set, [train_set_size, valid_set_size,test_set_size], generator=seed
)
# cpu core count workders
train_loader = DataLoader(
    train_set,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=min(os.cpu_count(),8),
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
    num_workers=min(os.cpu_count(),8),
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

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=min(os.cpu_count(),8),
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

############################### others
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.10, patience=3, verbose=False, mode="min"
)

tensorboard = pl_loggers.TensorBoardLogger(save_dir="exps")
trainer = pl.Trainer(
    default_root_dir="exps",
    callbacks=[early_stop_callback],
    logger=[tensorboard] if True else ["logger_0", "logger_1", "logger_2"],
    fast_dev_run=False,
    limit_train_batches=0.1,
    limit_val_batches=0.01,
    num_sanity_val_steps=2,
    max_epochs=-1,
)
    

# train model
# trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer = pl.Trainer()
trainer.test(model=autoencoder, dataloaders=test_loader)


