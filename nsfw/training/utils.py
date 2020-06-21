import numpy as np
from PIL import Image


def compute_dataset_stats(examples):
    mean = 0.
    var = 0.
    positive_count = 0
    for example_path, label in examples:
        image = np.array(Image.open(example_path).resize((224, 244))) / 255
        image = image[..., :3]

        mean += image.sum(axis=(0, 1))
        var += (image**2).sum(axis=(0, 1))
        positive_count += label

    dim = 224 * 224
    total_mean = mean / (len(examples) * dim)
    total_std = np.sqrt(var / (len(examples) * dim) * total_mean**2)
    positive_ratio = positive_count / len(examples)

    return {
        'mean': total_mean,
        'std': total_std,
        'positive_ratio': positive_ratio,
        'negative_ratio': 1. - positive_ratio
    }





