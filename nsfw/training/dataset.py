import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class NsfwDataset(Dataset):
    def __init__(self, image_paths, lables, transform=None):
        self._image_paths = image_paths
        self._labels = lables
        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index])
        label = self._labels[index]

        if self._transform:
            image = self._transform(image)
        else:
            image = to_tensor(image)


        return image, label