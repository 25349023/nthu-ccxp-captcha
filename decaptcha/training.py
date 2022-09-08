import os
import sys
from collections import abc
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from torch import nn
from torch.utils import data


class CaptchaNumberDataset(data.Dataset):
    def __init__(self, path_to_data_root='.', transform=None, train=True):
        self.data = np.load(os.path.join(path_to_data_root, 'images.npy'))
        self.labels = np.load(os.path.join(path_to_data_root, 'labels.npy'))

        if train:
            self.data = self.data[:-3000]
            self.labels = self.labels[:-3000]
        else:
            self.data = self.data[-3000:]
            self.labels = self.labels[-3000:]

        self.transform = transform

    def __getitem__(self, item):
        image, label = self.data[item], self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.data.shape[0]


def train_one_epoch(model, dataset, loss_fn, optim) -> float:
    losses = []
    for x, y in tqdm.tqdm(dataset, file=sys.stdout, desc='Training... ', leave=False):
        x, y = x.to('cuda'), y.to('cuda')
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    return np.array(losses).mean()


def test_one_epoch(model, dataset, loss_fn, metric: Optional[abc.Callable] = None):
    losses, metric_results = [], []
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataset, file=sys.stdout, desc='Testing... ', leave=False):
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            if metric:
                metric_results.append(metric(pred, y))

    if metric:
        metric_results = [np.array(metric_results).mean()]

    return np.array(losses).mean(), *metric_results


def fit(model, train_ld, test_ld, loss_fn, optim, metric=None, scheduler=None, epochs=20):
    for i in range(epochs):
        model.train()
        train_loss = train_one_epoch(model, train_ld, loss_fn, optim)
        model.eval()
        test_loss, *metric_results = test_one_epoch(model, test_ld, loss_fn, metric)

        print(f'Epoch {i:>2}: training loss = {train_loss:.6f}, testing loss = {test_loss:.6f}')
        if metric_results:
            m_str = f'\t| {metric.__name__} = {metric_results[0]:.6f}'
            print(m_str, end='\n\n')

        if scheduler is not None:
            scheduler.step()


def accuracy(pred: torch.Tensor, gt: torch.Tensor):
    return (pred.argmax(dim=1) == gt).sum().item() / pred.shape[0]


if __name__ == '__main__':
    train_dataset = CaptchaNumberDataset(transform=T.ToTensor(), train=True)
    test_dataset = CaptchaNumberDataset(transform=T.ToTensor(), train=False)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=64)

    tiny_net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).to('cuda')

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_net.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    epochs = 100

    fit(tiny_net, train_loader, test_loader, loss, optimizer, metric=accuracy, scheduler=scheduler, epochs=epochs)

    torch.save(tiny_net.state_dict(), 'tiny_net.pt')
