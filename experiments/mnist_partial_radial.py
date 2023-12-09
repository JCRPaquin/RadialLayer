import pytorch_lightning as pl

import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from radial_layer.model import PartialRadialLayer
from .data.mnist import MNISTDataModule


class PartialRadialLayerMNISTClassifier(pl.LightningModule):

    def __init__(self, lr_rate):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.rl1 = PartialRadialLayer(input_width=28*28, inner_width=8, depth=3)
        self.bn = nn.BatchNorm1d(8)
        self.act_fn = nn.GELU()
        self.rl2 = PartialRadialLayer(input_width=8, inner_width=10, depth=3)
        self.out_fn = nn.LogSoftmax()

        self.lr_rate = lr_rate

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        x = self.rl1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.rl2(x)
        x = self.out_fn(x)

        return x

    def eval_forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # inner layers (b, 1*28*28) -> (b, 10)
        x = self.rl1.eval_forward(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.rl2.eval_forward(x)
        x = self.out_fn(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        soft_logits = self.forward(x)
        soft_loss = self.cross_entropy_loss(soft_logits, y)
        hard_logits = self.eval_forward(x)
        hard_loss = self.cross_entropy_loss(hard_logits, y)

        loss = soft_loss + hard_loss

        self.log('train/loss', loss.detach().item())
        self.log('train/soft_loss', soft_loss.detach().item())
        self.log('train/hard_loss', hard_loss.detach().item())
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.eval_forward(x)
        loss = self.cross_entropy_loss(logits, y)

        distributions = torch.exp(logits)
        labels = torch.argmax(distributions, dim=1)
        accuracy = (labels == y).sum()/x.shape[0]

        self.log('val/hard_loss', loss.detach().item())
        self.log('val/accuracy', accuracy.detach().item())
        return {"loss": loss}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        self.log('test/loss', loss.detach().item())
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

if __name__ == "__main__":
    model = PartialRadialLayerMNISTClassifier(lr_rate=1e-3)
    data = MNISTDataModule()

    # Set Early Stopping
    early_stopping = EarlyStopping('val/hard_loss', mode='min', patience=5)
    # Log to wandb
    wandb_logger = WandbLogger(project="RadialLayer")
    wandb_logger.experiment.log_code(".")

    trainer = pl.Trainer(max_epochs=30, callbacks=[early_stopping],
                         logger=wandb_logger)
    trainer.fit(model, data)