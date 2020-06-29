from pathlib import Path

import torchvision.transforms as tvt
from PIL import Image

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
    'num_workers': 2,
    'batch_size': 32,
    'lr': 6e-4
}


def rgba_to_rgb(image):
    im = image[:3]
    if image.shape[0] != 3:
        im = image[0][np.newaxis]
        im = np.repeat(im, 3, axis=0)
    return im


# The if condition is necessary on Windows
if __name__ == '__main__':


    transforms = tvt.Compose([
        tvt.Resize((224, 224), interpolation=Image.BICUBIC),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Lambda(rgba_to_rgb),
        # tvt.Normalize(stats['mean'], stats['std'])
    ])

     # Multiple GPU training is not supported on Windows
    experiment = Experiment(checkpoint_path, config, transforms, use_gpus=[1])
    experiment.run()

