from PIL import Image

from nsfw.training.experiment import create_loaders, Experiment
from pathlib import Path
import torchvision.transforms as tvt


neutral_path = Path('/home/michal/data/zpi/images/neutral')
nsfw_path = Path('/home/michal/data/zpi/images/nsfw')
checkpoint_path = Path('/home/michal/data/zpi/checkpoints/v1')

transforms = tvt.Compose([
    tvt.Resize((224, 224), interpolation=Image.BICUBIC),
    tvt.ToTensor()
])
train_loader, val_loader = create_loaders(neutral_path, nsfw_path, transform=transforms)

experiment = Experiment(checkpoint_path)
experiment.run(train_loader, val_loader)

