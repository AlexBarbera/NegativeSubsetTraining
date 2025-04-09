import torch


class MNISTFCClassifier(torch.nn.Module):
    def __init__(self):
        self.network = torch.nn.Sequential([
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax()
        ])

    def forward(self, x):
        return self.network(x)