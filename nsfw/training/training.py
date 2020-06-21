import mlflow
import torch
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from nsfw.model.classifier import NsfwClassifier
from nsfw.training.dataset import NsfwDataset
from nsfw.data.utils import _scan_filenames, merge_filenames_and_labels


class NsfwSystem(LightningModule):
    def __init__(self, config, transforms):
        super().__init__()
        self._classifier = NsfwClassifier()
        self._loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1. - config[ratio]
                                 for ratio
                                 in ['negative_ratio', 'positive_ratio']]))
        self._transforms = transforms
        self._config = config

    def forward(self, x):
        return self._classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.forward(x)
        loss = self._loss(logits, y)

        mlflow.log_metric('train_loss', loss.item(), self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self._loss(logits, y)

        mlflow.log_metric('valid_loss', loss.item(), self.global_step)
        return {'valid_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        return {'valid_loss': avg_loss}

    def prepare_data(self) -> None:
        nsfw_filenames = _scan_filenames(self._config['nsfw_path'])
        neutral_filenames = _scan_filenames(self._config['neutral_path'])
        fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
        train, val = train_test_split(
            fnames_labels, shuffle=True, random_state=123
        )
        self._data = train, val

    def train_dataloader(self):
        return DataLoader(
            NsfwDataset(self._data[0], self._transforms),
            batch_size=self._config['batch_size'],
            shuffle=True,
            num_workers=self._config['num_workers']
        )

    def val_dataloader(self):
        return DataLoader(
            NsfwDataset(self._data[1], self._transforms),
            batch_size=self._config['batch_size'],
            shuffle=True,
            num_workers = self._config['num_workers']
        )

    def configure_optimizers(self):
        return Adam(self._classifier.parameters(), self._config['lr'])






