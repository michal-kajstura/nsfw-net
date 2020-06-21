from pathlib import Path

import torchvision.transforms as tvt
from PIL import Image

from nsfw.data.utils import _scan_filenames, merge_filenames_and_labels
from nsfw.training.experiment import Experiment
from nsfw.training.utils import compute_dataset_stats

neutral_path = Path(r'/home/michal/data/zpi/images/neutral')
nsfw_path = Path(r'/home/michal/data/zpi/images/nsfw')
checkpoint_path = Path(r'/home/michal/data/zpi/images/cp')



nsfw_filenames = _scan_filenames(nsfw_path)
neutral_filenames = _scan_filenames(neutral_path)
fnames_labels = merge_filenames_and_labels([nsfw_filenames, neutral_filenames], [1, 0])
stats = compute_dataset_stats(fnames_labels)
config = {
    'nsfw_path': nsfw_path,
    'neutral_path': neutral_path,
    'num_workers': 4,
    'batch_size': 2,
    'lr': 3e-4,
    **stats
}

transforms = tvt.Compose([
    tvt.Resize((224, 224), interpolation=Image.BICUBIC),
    tvt.RandomHorizontalFlip(),
    tvt.ToTensor(),
    tvt.Lambda(lambda img: img[:3]),
    tvt.Normalize(stats['mean'], stats['std'])
])
experiment = Experiment(checkpoint_path, config, transforms, use_gpus=[0, 1])
experiment.run()

