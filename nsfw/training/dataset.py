import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class NsfwDataset(Dataset):
    def __init__(self, examples, transform=None):
        self._examples = examples
        self._transform = transform

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, index):
        image, label = self._examples[index]
        image = Image.open(image)

        if self._transform:
            image = self._transform(image)
        else:
            image = to_tensor(image)

        return image, label