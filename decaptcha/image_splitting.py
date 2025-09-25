import pathlib
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def split(raw_img: np.ndarray, label: str, digits: int = 6) -> List[Tuple[np.ndarray, int]]:
    max_valid_w = 100
    raw_img = raw_img[:, :max_valid_w, :]
    h, w, c = raw_img.shape
    w_per_digit = w // digits
    split_points = list(range(0, w, w_per_digit))

    return [(raw_img[:, split_points[i]:split_points[i + 1]], int(lbl))
            for i, lbl in enumerate(label)]


def generate_dataset_from(src: pathlib.Path):
    img_pairs = []

    for path in src.glob('*.png'):
        label = path.stem[:6]
        raw_img = plt.imread(path)

        for (img, lbl) in split(raw_img, label):
            img_pairs.append((img[..., :3], lbl))

            if len(img_pairs) % 500 == 0:
                print(len(img_pairs), 'Proceeded.')

    random.shuffle(img_pairs)
    zipped_pairs = zip(*img_pairs)
    images, output_labels = zipped_pairs

    images = np.array(images)
    output_labels = np.array(output_labels, dtype=np.int64)
    print(images.shape, output_labels.shape)

    np.save('images.npy', images)
    np.save('labels.npy', output_labels)


if __name__ == '__main__':
    src_dir = pathlib.Path('./raw_data/')

    if not src_dir.exists():
        raise OSError(f"Source directory {src_dir} not found.")

    generate_dataset_from(src_dir)
