
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

# train with both splits
class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


# there are rich callbacks in pytorch lightning, check the docs for more
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.10, patience=3, verbose=False, mode="min"
)
# early_stop_callback_mine = MyEarlyStopping(
# monitor="val_accuracy", min_delta=0.10, patience=3, verbose=False, mode="max"
# )


tensorboard = pl_loggers.TensorBoardLogger(save_dir="exps")
# check lightning docs for more loggers like wandb, comet, neptune, mlflow

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
autoencoder = (
    LitAutoEncoder.load_from_checkpoint("/path/to/checkpoint.ckpt", **{})
    if False
    else autoencoder
)

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
try:
    trainer.fit(
        model=autoencoder,
        train_dataloaders=dataset.train_loader,
        val_dataloaders=dataset.valid_loader,
        ckpt_path=None if True else "some/path/to/my_checkpoint.ckpt",
    )
except Exception as e:
    print(e)
    traceback.print_exc()
    
train_set = MNIST(
    root="data/MNIST", download=True, train=True, transform=transforms.ToTensor()
)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(
    train_set, [train_set_size, valid_set_size], generator=seed
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


