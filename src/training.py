import  time
import argparse
from typing import List

import numpy
import torch
import torchvision
import sys
from src import models
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("epochs", type=int)
    output.add_argument("batch-size", type=int)
    output.add_argument("model", choices=["mnist", "cifar", "reduced-mnist", "reduced-cifar"])

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


def get_cifar():
    train = torchvision.datasets.CIFAR10("/tmp/cifar",
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                                               )
                                           ]
                                       ))

    test = torchvision.datasets.CIFAR10("/tmp/cifar", train=False,
                                       transform=torchvision.transforms.Compose(
                                           [
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                                               )
                                           ]
                                       ))

    return train, test


def get_index_of_outliers_iqr(model: torch.nn.Module, data: torch.utils.data.Dataset, loss_fn: torch.nn.Module,
                              factor: float = 1.5) -> List[int]:
    output = list()
    model.eval()

    with torch.no_grad():
        for x, y in data:
            yp = model(x)
            l = loss_fn(yp, y).detach().item()
            output.expand(l)

    m = numpy.median(output)
    iq1 = numpy.median(output[output < m])
    iq3 = numpy.median(output[output > m])
    iqr = (iq3 - iq1) * factor

    return numpy.where(output > iq3 + iqr)


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
        sw.add_scalar("loss/normal_mnist/train", train_loss, epoch)
        sw.add_scalar("time/normal_mnist/train", t, epoch)

        t = time.time()
        test_loss = test_loop(model, test, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/normal_mnist/test", test_loss, epoch)
        sw.add_scalar("time/normal_mnist/test", t, epoch)


def train_reduced_mnist(args):
    train, test = get_mnist()

    #train = torch.utils.data.Dataloader(train, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test = torch.utils.data.Dataloader(test, batch_size=args.batch_size, shuffle=True, num_workers=2)


    model = models.MNISTFCClassifier()
    optimizer = torch.optim.SGD(model.parameters())

    sw = SummaryWriter(args.tensorboard_path)

    for epoch in range(args.epochs):
        train = torch.utils.data.Subset(train, list(range(len(train))))
        t = time.time()
        train_loss = normal_train_loop(model, train, optimizer, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/reduced_mnist/train", train_loss, epoch)
        sw.add_scalar("time/reduced_mnist/train", t, epoch)

        t = time.time()
        train = torch.utils.data.Subset(train,
                                        get_index_of_outliers_iqr(model, train, torch.nn.NLLLoss(),
                                                                  args.iqr_factor)
                                        )
        t = time.time() - t
        sw.add_scalar("time/reduced_mnist/index", t, epoch)

        t = time.time()
        train_loss = normal_train_loop(model, train, optimizer, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/reduced_mnist/train_reduced", train_loss, epoch)
        sw.add_scalar("time/reduced_mnist/train_reduced", t, epoch)

        t = time.time()
        test_loss = test_loop(model, test, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/reduced_mnist/test", test_loss, epoch)
        sw.add_scalar("time/reduced_mnist/test", t, epoch)


def train_normal_cifar(args):
    train, test = get_cifar()

    train = torch.utils.data.Dataloader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test = torch.utils.data.Dataloader(test, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = models.CIFARClassifier()
    optimizer = torch.optim.SGD(model.parameters())

    sw = SummaryWriter(args.tensorboard_path)

    for epoch in range(args.epochs):
        t = time.time()
        train_loss = normal_train_loop(model, train, optimizer, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/normal_cifar/train", train_loss, epoch)
        sw.add_scalar("time/normal_cifar/train", t, epoch)

        t = time.time()
        test_loss = test_loop(model, test, torch.nn.NLLLoss())
        t = time.time() - t
        sw.add_scalar("loss/normal_cifar/test", test_loss, epoch)
        sw.add_scalar("time/normal_cifar/test", t, epoch)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.model == "mnist":
        train_normal_mnist(args)
    elif args.model == "cifar":
        train_normal_cifar(args)
    elif args.model == "reduced-mnist":
        train_reduced_mnist(args)
    elif args.model == "reduced-cifar":
        train_reduced_cifar(args)
