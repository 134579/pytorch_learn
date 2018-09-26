from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch

class LogisticRegression(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass


if __name__ == '__main__':
    data_dir = "./data"
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor)
    
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
