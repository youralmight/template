import os
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
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def any_lightning_module_function_or_hook(self):
        tensorboard_logger = self.logger.experiment
        fake_images = torch.Tensor(32, 3, 28, 28)
        tensorboard_logger.add_image("generated_images", fake_images, 0)


# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
autoencoder = (
    LitAutoEncoder.load_from_checkpoint("/path/to/checkpoint.ckpt", **{})
    if False
    else autoencoder
)
# check the checkpoint by myself


# # train model
# trainer = pl.Trainer()
# trainer.fit(model=autoencoder, train_dataloaders=dataset.train_loader)
# # test the model
# trainer = pl.Trainer()
# trainer.test(model=autoencoder, test_dataloaders=dataset.test_loader)

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
    monitor="val_accuracy", min_delta=0.10, patience=3, verbose=False, mode="max"
)
early_stop_callback_mine = MyEarlyStopping(
    monitor="val_accuracy", min_delta=0.10, patience=3, verbose=False, mode="max"
)


tensorboard = pl_loggers.TensorBoardLogger(save_dir="tb_logs")
# check lightning docs for more loggers like wandb, comet, neptune, mlflow

trainer = pl.Trainer(
    default_root_dir="exps",
    callbacks=[early_stop_callback],
    logger=[tensorboard] if True else ["logger_0", "logger_1", "logger_2"],
)

trainer.fit(
    model=autoencoder,
    train_dataloaders=dataset.train_loader,
    val_dataloaders=dataset.valid_loader,
    ckpt_path=None if True else "some/path/to/my_checkpoint.ckpt",
)
