import mlflow
import torch
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
from nsfw.model.classifier import NsfwClassifier
from nsfw.training.dataset import NsfwDataset
from nsfw.data.utils import _scan_filenames, merge_filenames_and_labels


class NsfwSystem(LightningModule):
    def __init__(self, config, transforms):
        super().__init__()
        self._classifier = NsfwClassifier()
        self._loss = nn.BCEWithLogitsLoss()
        self._transforms = transforms
        self._config = config

    def forward(self, x):
        return self._classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).float()
        logits = self.forward(x)
        loss = self._loss(logits, y)

        mlflow.log_metric('train_loss', loss.item(), self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).float()
        logits = self.forward(x)
        loss = self._loss(logits, y)
        preds = F.softmax(logits, dim=0) > 0.5
        return {'valid_loss': loss.detach(), 'true': y.detach(), 'preds': preds.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        true = torch.cat([x['true'] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()

        f1 = f1_score(true, preds)
        conf_mat = confusion_matrix(true, preds, normalize='all')
        tn, fp, fn, tp = conf_mat.ravel()
        mlflow.log_metrics({
            'valid_fscore': f1,
            'valid_loss': avg_loss.item(),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp

        }, step=self.global_step)
        return {'valid_loss': avg_loss}

    def prepare_data(self) -> None:
        nsfw_filenames = _scan_filenames(self._config['nsfw_path'])
        neutral_filenames = _scan_filenames(self._config['neutral_path'])
        fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
        train, val = train_test_split(
            fnames_labels, shuffle=True, random_state=123, test_size=0.1
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
        optimizer = Adam(self._classifier.parameters(), self._config['lr'])
        schedulder = StepLR(optimizer, step_size=4, gamma=0.1)
        return [optimizer], [schedulder]







