import itertools
from pathlib import Path
from typing import Tuple


def _scan_filenames(path: Path, extensions: Tuple[str, ...] = ('jpg', 'png')):
    return list(itertools.chain.from_iterable(
        path.rglob(f'*.{ext}') for ext in extensions
    ))

def merge_filenames_and_labels(filenames_list, labels_list):
    merged = []
    for fnames, label in zip(filenames_list, labels_list):
        merged.extend(zip(fnames, itertools.repeat(label)))
    return merged

