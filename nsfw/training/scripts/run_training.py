from pathlib import Path

import torchvision.transforms as tvt
from PIL import Image

from nsfw.data.utils import _scan_filenames, merge_filenames_and_labels
from nsfw.training.experiment import Experiment
from nsfw.training.utils import compute_dataset_stats

neutral_path = Path(r'D:\zpi-nsfw\data\instagram')
nsfw_path = Path(r'D:\zpi-nsfw\data\nsfw')
checkpoint_path = Path(r'D:\zpi-nsfw\data\checkpoints')

nsfw_filenames = _scan_filenames(nsfw_path)
neutral_filenames = _scan_filenames(neutral_path)
# fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
# stats = compute_dataset_stats(fnames_labels)
num_pos = len(nsfw_filenames)
num_neg = len(neutral_filenames)

config = {
    'nsfw_path': nsfw_path,
    'neutral_path': neutral_path,
    'num_workers': 4,
    'batch_size': 32,
    'lr': 1e-4,
    'positive_ratio': num_pos / (num_pos + num_neg),
    'negative_ratio': num_neg / (num_pos + num_neg)
}

def rgba_to_rgb(image):
    return image[:3]

transforms = tvt.Compose([
    tvt.Resize((224, 224), interpolation=Image.BICUBIC),
    tvt.RandomHorizontalFlip(),
    tvt.ToTensor(),
    tvt.Lambda(rgba_to_rgb),
    # tvt.Normalize(stats['mean'], stats['std'])
])

# The if condition is necessary on Windows
if __name__ == '__main__':
    # Multiple GPU training is not supported on Windows
    experiment = Experiment(checkpoint_path, config, transforms, use_gpus=[1])
    experiment.run()

