from typing import Tuple

import lightning.pytorch as pl

import torch
import wandb
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from radial_layer.nonlinearities import PairwisePowerProductPool
from radial_layer.model import PartialRadialLayer
from radial_layer.power_layer import PowerLayer
from experiments.data.mnist import MNISTDataModule


class MultiPartialRadialLayerMNISTClassifier(LightningModule):
    sub_0_rl1: PartialRadialLayer
    sub_1_rl1: PartialRadialLayer
    sub_2_rl1: PartialRadialLayer
    sub_3_rl1: PartialRadialLayer
    bn: nn.BatchNorm1d
    act_fn: nn.Module
    rl2: PartialRadialLayer
    out_fn: nn.LogSoftmax

    lr_rate: float

    def __init__(self,
                 learning_rate: float = 1e-3,
                 phase_change_epoch: int = 10,
                 layer1_depth: int = 3,
                 layer2_depth: int = 3,
                 spread_lambda: float = 1.,
                 quantile_lambda: float = 1.,
                 quantile_history_weight: float = 0.3,
                 load_balancing_lambda: float = 1.0,
                 **kwargs):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        def input_radial_layer():
            return torch.jit.script(PartialRadialLayer(
                input_width=28 * 28,
                inner_width=2,
                depth=layer1_depth,
                spread_lambda=spread_lambda,
                quantile_lambda=quantile_lambda,
                quantile_history_weight=quantile_history_weight,
                load_balancing_lambda=load_balancing_lambda))

        self.sub_0_rl1 = input_radial_layer()
        self.sub_1_rl1 = input_radial_layer()
        self.sub_2_rl1 = input_radial_layer()
        self.sub_3_rl1 = input_radial_layer()
        self.bn = nn.BatchNorm1d(8)
        self.nonlinearity = PairwisePowerProductPool(channel_width=2)
        self.act_fn = nn.GELU()

        self.rl2 = torch.jit.script(PartialRadialLayer(
            input_width=2 * 6,
            inner_width=10,
            depth=layer2_depth,
            spread_lambda=spread_lambda,
            quantile_lambda=quantile_lambda,
            quantile_history_weight=quantile_history_weight,
            load_balancing_lambda=load_balancing_lambda))
        self.out_fn = nn.LogSoftmax()

        self.lr_rate = learning_rate
        self.phase_change_epoch = phase_change_epoch
        self.phase_change_ready = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        sub_0_rl1_spread_loss = self.sub_0_rl1.spread_loss(x)
        sub_1_rl1_spread_loss = self.sub_1_rl1.spread_loss(x)
        sub_2_rl1_spread_loss = self.sub_2_rl1.spread_loss(x)
        sub_3_rl1_spread_loss = self.sub_3_rl1.spread_loss(x)
        rl1_spread_loss = sub_0_rl1_spread_loss + sub_1_rl1_spread_loss + \
            sub_2_rl1_spread_loss + sub_3_rl1_spread_loss
        x = torch.hstack([
            self.sub_0_rl1(x),
            self.sub_1_rl1(x),
            self.sub_2_rl1(x),
            self.sub_3_rl1(x)
        ])
        x = self.bn(x)
        x = x.view(batch_size, 4, -1)
        x = self.nonlinearity(x)
        x = self.act_fn(x)
        rl2_spread_loss = self.rl2.spread_loss(x)
        x = self.rl2(x)
        x = self.out_fn(x)

        return x, rl1_spread_loss + rl2_spread_loss

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        x = torch.hstack([
            self.sub_0_rl1.eval_forward(x),
            self.sub_1_rl1.eval_forward(x),
            self.sub_2_rl1.eval_forward(x),
            self.sub_3_rl1.eval_forward(x)
        ])
        x = self.bn(x)
        x = x.view(batch_size, 4, -1)
        x = self.nonlinearity(x)
        x = self.act_fn(x)
        x = self.rl2.eval_forward(x)
        x = self.out_fn(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def tree_loss(self):
        return self.sub_0_rl1.tree_loss() + self.sub_1_rl1.tree_loss() + \
            self.sub_2_rl1.tree_loss() + self.sub_3_rl1.tree_loss() + self.rl2.tree_loss()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        soft_logits, spread_loss = self.forward(x)
        soft_loss = self.cross_entropy_loss(soft_logits, y)
        hard_logits = self.eval_forward(x)
        hard_loss = self.cross_entropy_loss(hard_logits, y)
        tree_loss = self.tree_loss()

        if self.current_epoch > self.phase_change_epoch:
            loss = soft_loss + hard_loss
        else:
            loss = soft_loss + hard_loss + spread_loss + tree_loss

        self.log('train/loss', loss.detach().item())
        self.log('train/soft_loss', soft_loss.detach().item())
        self.log('train/hard_loss', hard_loss.detach().item())
        self.log('train/spread_loss', spread_loss.detach().item())
        self.log('train/tree_loss', tree_loss.detach().item())

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self.eval_forward(x)
        loss = self.cross_entropy_loss(logits, y)

        distributions = torch.exp(logits)
        labels = torch.argmax(distributions, dim=1)
        accuracy = (labels == y).sum() / x.shape[0]

        self.log('val/hard_loss', loss.detach().item(), sync_dist=True)
        self.log('val/accuracy', accuracy.detach().item(), sync_dist=True)

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
        return [optimizer], []  # [lr_scheduler]


if __name__ == "__main__":
    # 2e-3 is too high
    # 1e-3 seems fine
    # 1e-4 is too low
    model = MultiPartialRadialLayerMNISTClassifier(learning_rate=1e-3, phase_change_epoch=3)
    data = MNISTDataModule()

    # Set Early Stopping
    early_stopping = EarlyStopping('val/hard_loss', mode='min', patience=10)
    # Log to wandb
    wandb_logger = WandbLogger(project="RadialLayer", group="MultiRadial-Nonlinearities")
    wandb_logger.experiment.log_code(".")

    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping],
                         logger=wandb_logger, accelerator="cpu")
    trainer.fit(model, data)
