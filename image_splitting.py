import pathlib

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def split(raw_img: np.ndarray, label: str, digits: int = 6) -> List[Tuple[np.ndarray, int]]:
    h, w, c = raw_img.shape
    w_per_digit = w // digits
    split_points = list(range(0, w, w_per_digit))

    return [(raw_img[:, split_points[i]:split_points[i + 1]], int(lbl))
            for i, lbl in enumerate(label)]


def generate_dataset_from(src: pathlib.Path, dest: pathlib.Path):
    file_idx = 0
    output_labels = []

    for path in src.glob('*.png'):
        label = path.stem[:6]
        raw_img = plt.imread(path)

        for (img, lbl) in split(raw_img, label):
            plt.imsave(dest / f'img_{file_idx}.png', img)
            output_labels.append(lbl)
            file_idx += 1

            if file_idx % 500 == 0:
                print(file_idx, 'Proceeded.')

    output_labels = np.array(output_labels, dtype=int)
    np.save('labels.npy', output_labels)


if __name__ == '__main__':
    src_dir = pathlib.Path('./raw_data/')
    dest_dir = pathlib.Path('./data/')

    if not src_dir.exists():
        raise OSError(f"Source directory {src_dir} not found.")

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    generate_dataset_from(src_dir, dest_dir)
