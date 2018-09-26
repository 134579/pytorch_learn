from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pylab as plt

class LogisticRegression(torch.nn.Module):

    def __init__(self, in_dimen, out_dimen):
        super().__init__()

        self.logistic = nn.Linear(in_dimen, out_dimen)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        z1 = self.logistic(x)
        z2 = self.sigmoid(z1)
        z3 = self.softmax(z2)
        return z3


if __name__ == '__main__':
    data_dir = "./data"
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())

    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LogisticRegression()


    plt.imshow(train_dataset[0][0].view(28,28))
