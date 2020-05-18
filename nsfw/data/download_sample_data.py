import os
import random
import sys
from io import BytesIO
from pathlib import Path
import uuid
from urllib.parse import urlparse

import requests
import PIL.Image as Image
from urllib3.exceptions import MaxRetryError


def download_data(url, save_location, sample=10):
    save_location_path = Path(save_location)
    save_location_path.mkdir(parents=True, exist_ok=True)

    image_urls = requests.get(url).content
    image_urls = image_urls.split()
    image_urls = [u.decode('utf-8') for u in image_urls]

    random.shuffle(image_urls)

    for img_url in image_urls[:sample]:
        try:
            image = _download_image(img_url)
            image_url_path = Path(urlparse(img_url).path)
            filename = Path(save_location_path, image_url_path.name)
            image.save(filename)
        except (ConnectionError, MaxRetryError):
            print(f"Cannot download this image: {img_url}", file=sys.stderr)
            continue

def _download_image(url) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise ConnectionError

