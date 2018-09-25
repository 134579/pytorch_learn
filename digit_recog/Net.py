from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        """
        see https://pytorch.org/tutorials/_images/mnist.png
        """
        super(Net, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(1, 6, 5)  # in: 32x32x1 out: 28x28x6

        self.relu1 = nn.ReLU()
        # pool1
        self.pool1 = nn.MaxPool2d(2)

        # conv2
        self.conv2 = nn.Conv2d(6, 16, 5)  # in: 14x14x6 out: 10x10x16

        # pool2
        self.pool2 = nn.MaxPool2d(2)

        # full connect 1
        self.fc1 = nn.Linear(5 * 5 * 16, 120)  # in: 5x5x16 out: 120

        # full connect 2
        self.fc2 = nn.Linear(120, 84, bias=True)

        # full connect 3
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # type: torch.Tensor
        x = x.view(-1, Net.num_flat_features(x))  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x: torch.Tensor) -> int:
        dims = x.size()[1:]
        return torch.prod(torch.tensor(tuple(dims)))


