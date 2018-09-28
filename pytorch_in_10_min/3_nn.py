from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt

from misc import tic, toc


class NeuralNet(torch.nn.Module):
    
    def __init__(self, in_dimen, out_dimen):
        super().__init__()
        n_hidden_1 = 300
        n_hidden_2 = 100
        n_hidden_3 = 100
        
        self.layer1 = nn.Linear(in_dimen, n_hidden_1)
        self.layer2 = nn.Sigmoid()
        self.layer3 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer4 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer5 = nn.Linear(n_hidden_3, out_dimen)
    
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)
        x = self.layer5.forward(x)
        return x


class Trainer:
    def __init__(self, model: nn.Module, loss: nn.Module, optimizer: optim.Optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        pass
    
    def train(self, num_epoches: int):
        pass


if __name__ == '__main__':
    data_dir = "./data"
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())
    
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = NeuralNet(28 * 28, 10)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epoches = 100
    use_gpu = True
    if use_gpu:
        model.cuda()
    for epoch in range(num_epoches):
        running_loss = 0.0
        tic()
        for i, data in enumerate(train_loader, 1):
            # [n,1,28,28], [1000]
            img, label = data  # type: torch.Tensor,torch.Tensor
            
            img = img.view(img.size(0), -1)  # 将图片展开成 28x28
            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            
            optimizer.zero_grad()
            
            out = model(img)
            loss = criterion(out, label)
            
            # loss1 = sum([-out[j][label[j]] + torch.log(torch.sum(torch.exp(out[j]))) for j in range(1000)])
            # print(loss, loss1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                # accuracy
                _, predict = out.max(1)
                num_correct = (predict == label).sum()
                accuracy = num_correct.item() * 1.0 / label.size(0)
                print(f'[{epoch + 1:d}, {i*batch_size:5d}] loss: {running_loss:.3f}, acc: {accuracy}')
                running_loss = 0.0
        toc()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)  # 将图片展开成 28x28
            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
