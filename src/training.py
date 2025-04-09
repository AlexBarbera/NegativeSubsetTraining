import  time
import argparse

import torch
import torchvision
import sys
from src import models
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("epochs", type=int)
    output.add_argument("batch-size", type=int)

    return output.parse_args(sys.argv[1:])


def get_mnist():
    train = torchvision.datasets.MNIST("/tmp/mnist",
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor()
                                           ]
                                       ))

    test = torchvision.datasets.MNIST("/tmp/mnist", train=False,
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor()
                                           ]
                                       ))

    return train, test


def normal_train_loop(model, dataloader, optimizer, loss_fn):
    model.train()
    l = list()
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        yp = model(x)
        loss = loss_fn(yp, y)
        loss.backward()
        optimizer.step()
        l.append(loss.item().detach())

    l = sum(l) / len(l)

    return l


def test_loop(model, dataloader, loss_fn):
    model.eval()

    l = list()
    for batch_idx, (x, y) in enumerate(dataloader):
        yp = model(x)
        loss = loss_fn(yp, y)
        l.append(loss.item().detach())

    l = sum(l) / len(l)

    return l


def train_normal_mnist(args):
    train, test = get_mnist()

    train = torch.utils.data.Dataloader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test = torch.utils.data.Dataloader(test, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = models.MNISTFCClassifier()
    optimizer = torch.optim.SGD(model.parameters())

    sw = SummaryWriter(args.tensorboard_path)

    for epoch in range(args.epochs):
        t = time.time()
        train_loss = normal_train_loop(model, train, optimizer, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/normal/train", train_loss, epoch)
        sw.add_scalar("time/normal/train", t, epoch)

        t = time.time()
        test_loss = test_loop(model, test, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/normal/test", train_loss, epoch)
        sw.add_scalar("time/normal/test", t, epoch)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train_normal_mnist(args)
