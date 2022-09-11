import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from torch import nn

BASE_URL = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'


def get_img_src() -> str:
    res = requests.get(BASE_URL)
    soup = BeautifulSoup(res.text, 'lxml')
    img = soup.select_one('.input_box + img')
    return img['src']


def download_img(src: str) -> np.ndarray:
    res = requests.get(BASE_URL + src)
    f = io.BytesIO(res.content)
    img = plt.imread(f)
    return img


def split(raw_img: np.ndarray, digits: int = 6) -> List[np.ndarray]:
    h, w, c = raw_img.shape
    w_per_digit = w // digits
    split_points = list(range(0, w, w_per_digit))

    return [raw_img[:, split_points[i]:split_points[i + 1]]
            for i in range(digits)]


def load_tiny_net(ckpt_path: str):
    tiny_net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.Linear(64, 10)
    )

    tiny_net.load_state_dict(torch.load(ckpt_path))
    tiny_net.eval()
    return tiny_net


def decaptcha_one():
    src = get_img_src()
    raw_img = download_img(src)
    img_patches = split(raw_img[..., :3])

    plt.imshow(raw_img)
    plt.show()

    tiny_net = load_tiny_net('tiny_net.pt')
    for img in img_patches:
        predict = tiny_net(torch.tensor(img[np.newaxis]).permute(0, 3, 1, 2))
        print(predict.argmax(dim=1).item(), end='')
    print()


if __name__ == '__main__':
    decaptcha_one()
