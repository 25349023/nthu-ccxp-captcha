import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'


def get_pwd_str() -> str:
    res = requests.get(BASE_URL)
    soup = BeautifulSoup(res.text, 'lxml')
    img = soup.select_one('.input_box + img')
    return img['src']


def manually_label(src: str) -> str:
    res = requests.get(BASE_URL + src)
    f = io.BytesIO(res.content)
    img = plt.imread(f)
    plt.imshow(img)
    plt.show()
    label = input('input the numbers you just saw: ')
    return label


def collect_one(save_dir: Path, generate_count: int):
    src = get_pwd_str()
    file_prefix = manually_label(src)

    for i in range(generate_count):
        res = requests.get(BASE_URL + src)
        with open(save_dir / f'{file_prefix}_{i}.png', 'wb') as f:
            f.write(res.content)


def collect_many(save_dir: Path, n_round: int, cnt_per_round: int):
    for r in range(n_round):
        collect_one(save_dir, cnt_per_round)


if __name__ == '__main__':
    dire = Path('./raw_data/')
    if not dire.exists():
        dire.mkdir(parents=True)

    collect_many(dire, 42, 50)
