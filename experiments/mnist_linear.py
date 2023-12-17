from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from experiments.data.mnist import MNISTDataModule
from radial_layer.power_layer import PowerLayer


class LinearMNISTClassifier(pl.LightningModule):

    linear1: nn.Linear
    bn: nn.BatchNorm1d
    act_fn: nn.Module
    linear2: nn.Linear
    out_fn: nn.LogSoftmax

    lr_rate: float

    def __init__(self, lr_rate: float, phase_change_epoch: int = 10):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.linear1 = nn.Linear(28*28, 8)
        self.bn = nn.BatchNorm1d(8)
        self.act_fn = nn.GELU()
        self.power_layer = PowerLayer(input_width=8, power=4)
        self.linear2 = nn.Linear(8*4, 10)
        self.out_fn = nn.LogSoftmax()

        self.lr_rate = lr_rate
        self.phase_change_epoch = phase_change_epoch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.power_layer(x)
        x = self.linear2(x)
        x = self.out_fn(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        soft_logits = self.forward(x)
        loss = self.cross_entropy_loss(soft_logits, y)

        self.log('train/loss', loss.detach().item())

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        distributions = torch.exp(logits)
        labels = torch.argmax(distributions, dim=1)
        accuracy = (labels == y).sum()/x.shape[0]

        self.log('val/loss', loss.detach().item())
        self.log('val/accuracy', accuracy.detach().item())

        return {"loss": loss}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        self.log('test/loss', loss.detach().item())
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_rate, weight_decay=0.01)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

if __name__ == "__main__":
    model = LinearMNISTClassifier(lr_rate=1e-3, phase_change_epoch=-1)
    data = MNISTDataModule()

    # Set Early Stopping
    early_stopping = EarlyStopping('val/loss', mode='min', patience=10)
    # Log to wandb
    wandb_logger = WandbLogger(project="RadialLayer", group="Linear Baseline")
    wandb_logger.experiment.log_code(".")

    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping],
                         logger=wandb_logger)
    trainer.fit(model, data)
