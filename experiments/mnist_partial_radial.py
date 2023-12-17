from typing import Tuple

import pytorch_lightning as pl

import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from radial_layer.model import PartialRadialLayer
from radial_layer.power_layer import PowerLayer
from experiments.data.mnist import MNISTDataModule


class PartialRadialLayerMNISTClassifier(pl.LightningModule):

    rl1: PartialRadialLayer
    bn: nn.BatchNorm1d
    act_fn: nn.Module
    rl2: PartialRadialLayer
    out_fn: nn.LogSoftmax

    lr_rate: float

    def __init__(self,
                 learning_rate: float,
                 phase_change_epoch: int = 10,
                 layer1_depth: int = 3,
                 layer2_depth: int = 3,
                 spread_lambda: float = 10.,
                 quantile_lambda: float = 5.,
                 max_power: int = 4):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.rl1 = torch.jit.script(PartialRadialLayer(
            input_width=28*28,
            inner_width=8,
            depth=layer1_depth,
            spread_lambda=spread_lambda))
        self.rl1.a_i.requires_grad=True
        self.rl1.b_i.requires_grad=True
        self.rl1.w_i.requires_grad=True
        self.rl1.quantile_lambda = quantile_lambda
        self.bn = nn.BatchNorm1d(8)
        self.act_fn = nn.GELU()
        self.power_layer = PowerLayer(input_width=8, power=max_power)
        self.rl2 = torch.jit.script(PartialRadialLayer(
            input_width=8*max_power,
            inner_width=10,
            depth=layer2_depth,
            spread_lambda=spread_lambda))
        self.rl2.a_i.requires_grad=True
        self.rl2.b_i.requires_grad=True
        self.rl2.w_i.requires_grad=True
        self.rl2.quantile_lambda = quantile_lambda
        self.out_fn = nn.LogSoftmax()

        self.lr_rate = learning_rate
        self.phase_change_epoch = phase_change_epoch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        rl1_spread_loss = self.rl1.spread_loss(x)
        x = self.rl1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.power_layer(x)
        rl2_spread_loss = self.rl2.spread_loss(x)
        x = self.rl2(x)
        x = self.out_fn(x)

        return x, rl1_spread_loss + rl2_spread_loss

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        x = self.rl1.eval_forward(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.power_layer(x)
        x = self.rl2.eval_forward(x)
        x = self.out_fn(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        soft_logits, spread_loss = self.forward(x)
        soft_loss = self.cross_entropy_loss(soft_logits, y)
        hard_logits = self.eval_forward(x)
        hard_loss = self.cross_entropy_loss(hard_logits, y)
        tree_loss = self.rl1.tree_loss() + self.rl2.tree_loss()

        if self.current_epoch > self.phase_change_epoch:
            loss = soft_loss + hard_loss
        else:
            loss = soft_loss + hard_loss + spread_loss + tree_loss

        self.log('train/loss', loss.detach().item())
        self.log('train/soft_loss', soft_loss.detach().item())
        self.log('train/hard_loss', hard_loss.detach().item())
        self.log('train/spread_loss', spread_loss.detach().item())
        self.log('train/tree_loss', tree_loss.detach().item())

        node_splits = F.sigmoid(self.rl1.b_i)/(0.5+F.sigmoid(self.rl1.w_i))
        for i in range(node_splits.shape[1]):
            self.log(f'train/node_split_{i}', node_splits[0][i].detach().item())

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        buckets = self.rl1.scaled_distribution(x.view(-1, 28*28))
        buckets = torch.argmax(buckets, dim=1)

        angle_dist = self.rl1.angles(x.view(-1, 28*28)).unsqueeze(-1)

        logits = self.eval_forward(x)
        loss = self.cross_entropy_loss(logits, y)

        distributions = torch.exp(logits)
        labels = torch.argmax(distributions, dim=1)
        accuracy = (labels == y).sum()/x.shape[0]

        self.log('val/hard_loss', loss.detach().item())
        self.log('val/accuracy', accuracy.detach().item())
        self.logger.experiment.log({
            'val/rl1_dist_plot': wandb.Image(self.rl1.plot_distribution().T),
            'val/rl1_angle_plot': wandb.Histogram(angle_dist.view(-1))
        })

        bucket_totals = dict()
        for i in range(x.shape[0]):
            bucket = int(buckets[i].item())
            if bucket in bucket_totals:
                bucket_totals[bucket] += 1
            else:
                bucket_totals[bucket] = 1

        for i in range(2**self.rl1.depth):
            self.log(f'val/total_bucket_{i}', bucket_totals.get(i, 0))

        for i in range(self.rl1.quantiles.shape[-1]):
            self.log(f'val/rl1_quantile_{i}', self.rl1.quantiles[0,i])

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
    # 2e-3 is too high
    # 1e-3 seems fine
    # 1e-4 is too low
    model = PartialRadialLayerMNISTClassifier(learning_rate=1e-3, phase_change_epoch=3)
    data = MNISTDataModule()

    # Set Early Stopping
    early_stopping = EarlyStopping('val/hard_loss', mode='min', patience=10)
    # Log to wandb
    wandb_logger = WandbLogger(project="RadialLayer")
    wandb_logger.experiment.log_code(".")

    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping],
                         logger=wandb_logger)
    trainer.fit(model, data)
