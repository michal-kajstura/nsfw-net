import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import mlflow

class Trainer:
    def __init__(self, model: nn.Module,
                 optimizer:  Optimizer,
                 loss_function: nn.Module = nn.CrossEntropyLoss(),
                 scheduler=None,
                 device=torch.device('cpu')):
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._scheduler = scheduler
        self._device = device
        self._step = 0

    def train(self, train_loader, validation_loader=None, epochs=5):
        for epoch in range(epochs):
            self._train_loop(train_loader)

            if validation_loader:
                self._validation_loop(validation_loader)



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
        mlflow.log_metric('validation_loss', average_loss, self._step)

