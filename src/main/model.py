import traceback
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self,fake_argument):
        super().__init__()
        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fake_argument=fake_argument

    def on_train_start(self) -> None:
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def on_train_end(self) -> None:
        return super().on_train_end()

    def on_validation_epoch_start(self) -> None:
        self.epoch_batch_loss_list=[]
        return super().on_validation_epoch_start()

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("batch_val_loss", val_loss)
        self.epoch_batch_loss_list.append(val_loss)

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return super().on_validation_batch_end(
            outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(self) -> None:
        # do something with the outputs of all validation steps
        # e.g. calculate mean of all losses
        # pdb.set_trace()
        mean_loss = torch.stack(self.epoch_batch_loss_list).mean()
        self.log("val_loss", mean_loss)
        del self.epoch_batch_loss_list

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def any_lightning_module_function_or_hook(self):
        tensorboard_logger = self.logger.experiment
        fake_images = torch.Tensor(32, 3, 28, 28)
        tensorboard_logger.add_image("generated_images", fake_images, 0)





