from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import mlflow

from nsfw.model.classifier import NsfwClassifier


class Trainer:
    def __init__(self, model: nn.Module,
                 optimizer:  Optimizer,
                 loss_function: nn.Module = nn.CrossEntropyLoss(),
                 scheduler=None,
                 device=torch.device('cpu'),
                 checkpoint_path: Optional[Path] = None):
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._scheduler = scheduler
        self._device = device
        self._step = 0
        self._checkpoint_path = checkpoint_path

    def train(self, train_loader, validation_loader=None, epochs=5):
        if self._checkpoint_path:
            self._checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self._train_loop(train_loader)

            if validation_loader:
                loss = self._validation_loop(validation_loader)
                mlflow.log_metric('validation_loss', loss, self._step)
                if self._checkpoint_path:
                    self._save_checkpoint(Path(self._checkpoint_path, f'{epoch}.pth'))

    def _train_loop(self, loader):
        self._model.train()
        for images, labels in tqdm(loader):
            images = images.to(self._device)
            labels = labels.to(self._device)

            logits = self._model(images)
            loss = self._loss_function(logits, labels)
            mlflow.log_metric('train_loss', loss.item(), self._step)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()


    def _validation_loop(self, loader):
        self._model.eval()
        accumulated_loss = 0.
        with torch.no_grad():
            for images, labels in tqdm(loader):
                images = images.to(self._device)
                labels = labels.to(self._device)

                logits = self._model(images)
                loss = self._loss_function(logits, labels)

                accumulated_loss += loss.item()

        average_loss = accumulated_loss / len(loader)
        return average_loss

    def _save_checkpoint(self, checkpoint_path: Path):
        checkpoint_path.touch(exist_ok=True)
        torch.save({
            'step': self._step,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }, checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._step = checkpoint['step']

