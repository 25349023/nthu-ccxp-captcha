import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def split(raw_img: np.ndarray, label: str, digits: int = 6) -> List[Tuple[np.ndarray, int]]:
    h, w, c = raw_img.shape
    w_per_digit = w // digits
    split_points = list(range(0, w, w_per_digit))

    return [(raw_img[:, split_points[i]:split_points[i + 1]], int(lbl))
            for i, lbl in enumerate(label)]


def generate_dataset_from(src: pathlib.Path):
    images = []
    output_labels = []

    for path in src.glob('*.png'):
        label = path.stem[:6]
        raw_img = plt.imread(path)

        for (img, lbl) in split(raw_img, label):
            images.append(img[..., :3])
            output_labels.append(lbl)

            if len(images) % 500 == 0:
                print(len(images), 'Proceeded.')

    images = np.array(images)
    output_labels = np.array(output_labels, dtype=int)
    np.save('images.npy', images)
    np.save('labels.npy', output_labels)


if __name__ == '__main__':
    src_dir = pathlib.Path('./raw_data/')

    if not src_dir.exists():
        raise OSError(f"Source directory {src_dir} not found.")

    generate_dataset_from(src_dir)
