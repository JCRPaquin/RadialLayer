import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(pl.LightningDataModule):
    mnist_test: MNIST
    mnist_predict: MNIST
    mnist_train: Subset
    mnist_val: Subset

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.0,), (1.0,))])

        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=transform)
        self.mnist_predict = MNIST(self.data_dir, train=False, download=True, transform=transform)
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)