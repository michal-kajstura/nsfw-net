import itertools
from sklearn.model_selection import train_test_split
from pathlib import Path
import torchvision.transforms as tfms

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from nsfw.model.classifier import NsfwClassifier
from nsfw.training.dataset import NsfwDataset
from nsfw.training.trainer import Trainer
import mlflow


class Experiment:
    def __init__(self):
        model = NsfwClassifier()
        optimizer = Adam(model.parameters(), lr=3e-4)
        loss = nn.CrossEntropyLoss()
        self._trainer = Trainer(model, optimizer, loss)

    def run(self, train_loader: DataLoader, validation_loader: DataLoader=None):
        with mlflow.start_run():
            self._trainer.train(train_loader, validation_loader)


def create_loaders(path):
    directories_to_classes = {
        'nsfw': 'nsfw',
        'neutral': 'neutral'
    }

    nsfw_filenames = _scan_filenames(Path(path, directories_to_classes['nsfw']))
    neutral_filenames = _scan_filenames(Path(path, directories_to_classes['nsfw']))

    fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
    x, y = zip(*fnames_labels)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=True, random_state=123
    )

    transform = tfms.Compose([
        tfms.Resize((224, 224)),
        tfms.ToTensor(),
    ])
    train_loader = DataLoader(
        NsfwDataset(x_train, y_train, transform=transform),
        shuffle=True, batch_size=4,
    )
    val_loader = DataLoader(
        NsfwDataset(x_val, y_val, transform=transform),
        shuffle=True, batch_size=4,
    )

    return train_loader, val_loader


def _scan_filenames(path):
    return list(path.glob('*'))

def merge_filenames_and_labels(filenames_list, labels_list):

    merged = []
    for fnames, label in zip(filenames_list, labels_list):
        merged.extend(zip(fnames, itertools.repeat(label)))
    return merged


create_loaders('/home/michal/data/zpi/images')
