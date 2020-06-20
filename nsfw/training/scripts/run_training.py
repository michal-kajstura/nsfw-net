import torch
from PIL import Image

from nsfw.training.experiment import create_loaders, Experiment
from pathlib import Path
import torchvision.transforms as tvt

neutral_path = Path(r'D:\zpi-nsfw\data\instagram')
nsfw_path = Path(r'D:\zpi-nsfw\data\nsfw')
checkpoint_path = Path(r'D:\zpi-nsfw\data\checkpoints\v1')

transforms = tvt.Compose([
    tvt.Resize((224, 224), interpolation=Image.BICUBIC),
    tvt.ToTensor(),
    tvt.Lambda(lambda img: img[:3])
])
train_loader, val_loader = create_loaders(neutral_path, nsfw_path, transform=transforms)

experiment = Experiment(checkpoint_path, device=torch.device('cuda:1'))
experiment.run(train_loader, val_loader)

