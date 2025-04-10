import torch
import torch.nn.functional as F


class MNISTFCClassifier(torch.nn.Module):
    def __init__(self):
        super(MNISTFCClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax()
        )

    def forward(self, x):
        return self.network(x)


class CIFARClassifier(torch.nn.Module):  # from pytorch example
    def __init__(self):
        super(CIFARClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x