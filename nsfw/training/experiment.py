import itertools
from itertools import chain
from pathlib import Path
from typing import Tuple, Optional, Callable

import mlflow
import torch
import torchvision.transforms as tfms
from PIL.Image import Image
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from nsfw.model.classifier import NsfwClassifier
from nsfw.training.dataset import NsfwDataset
from nsfw.training.trainer import Trainer


class Experiment:
    def __init__(self, checkpoint_path: Path, device: torch.device = torch.device('cpu')):
        model = NsfwClassifier()
        optimizer = Adam(model.parameters(), lr=3e-4)
        loss = nn.CrossEntropyLoss()

        self._trainer = Trainer(model, optimizer, loss,
                                device=device,
                                checkpoint_path=checkpoint_path)

    def run(self, train_loader: DataLoader, validation_loader: DataLoader=None):
        with mlflow.start_run():
            self._trainer.train(train_loader, validation_loader)


def create_loaders(neutral_images_path: Path, nsfw_images_path: Path,
                   transform: Optional[Callable[[Image], Tensor]] = None)\
        -> Tuple[DataLoader, DataLoader]:
    nsfw_filenames = _scan_filenames(nsfw_images_path)
    neutral_filenames = _scan_filenames(neutral_images_path)

    fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
    x, y = zip(*fnames_labels)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=True, random_state=123
    )

    train_loader = DataLoader(
        NsfwDataset(x_train, y_train, transform=transform),
        shuffle=True, batch_size=4,
    )
    val_loader = DataLoader(
        NsfwDataset(x_val, y_val, transform=transform),
        shuffle=True, batch_size=4,
    )

    return train_loader, val_loader


def _scan_filenames(path: Path, extensions: Tuple[str, ...] = ('jpg', 'png')):
    return list(chain.from_iterable(
        path.rglob(f'*.{ext}') for ext in extensions
    ))

def merge_filenames_and_labels(filenames_list, labels_list):
    merged = []
    for fnames, label in zip(filenames_list, labels_list):
        merged.extend(zip(fnames, itertools.repeat(label)))
    return merged


