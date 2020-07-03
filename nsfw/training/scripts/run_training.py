from pathlib import Path

import torchvision.transforms as tvt
from PIL import Image
import sys
sys.path.append(r'D:\zpi-nsfw\nsfw-net')

from nsfw.data.utils import _scan_filenames, merge_filenames_and_labels
from nsfw.training.experiment import Experiment
from nsfw.training.utils import compute_dataset_stats
import numpy as np


neutral_path = Path(r'D:\zpi-nsfw\data\instagram')
nsfw_path = Path(r'D:\zpi-nsfw\data\nsfw')
checkpoint_path = Path(r'D:\zpi-nsfw\data\checkpoints')

config = {
    'nsfw_path': nsfw_path,
    'neutral_path': neutral_path,
    'num_workers': 3,
    'batch_size': 22,
    'learning_rates': {
        'layer1': 1e-6,
        'layer2': 1e-5,
        'layer3': 1e-4,
        'layer4': 3e-4,
        'fc': 7e-4
    }
}


def rgba_to_rgb(image):
    im = image[:3]
    if image.shape[0] != 3:
        im = image[0][np.newaxis]
        im = np.repeat(im, 3, axis=0)
    return im


# The if condition is necessary on Windows
if __name__ == '__main__':

    train_transforms = tvt.Compose([
        tvt.Resize((224, 224), interpolation=Image.BICUBIC),
        tvt.RandomHorizontalFlip(),
        tvt.RandomGrayscale(p=0.05),
        tvt.ToTensor(),
        tvt.RandomErasing(p=0.1, scale=(0.02, 0.2)),
        tvt.Lambda(rgba_to_rgb),
        tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    val_transforms = tvt.Compose([
        tvt.Resize((224, 224), interpolation=Image.BICUBIC),
        tvt.ToTensor(),
        tvt.Lambda(rgba_to_rgb),
        tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transforms = train_transforms, val_transforms
     # Multiple GPU training is not supported on Windows
    experiment = Experiment(checkpoint_path, config, transforms, use_gpus=[1])
    experiment.run()

