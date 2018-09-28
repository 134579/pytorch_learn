import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt

from misc import tic, toc


class CNN(torch.nn.Module):
    def __init__(self, in_dimen, out_dimen):
        """
        :param in_dimen: (channel_in, height, width)
        :param out_dimen: nb_class
        """
        super().__init__()
        self.in_dimen = in_dimen
        channel_in, height, width = in_dimen
        self.out_dimen = out_dimen
        
        kernel_size1 = 4
        channel_out1 = 10
        
        channel_out2 = 20
        kernel_size2 = 4
        
        conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out1, kernel_size=kernel_size1)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d((2, 2))
        conv2 = nn.Conv2d(channel_out1, channel_out2, kernel_size2)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d((2, 2))
        
        self.layers = nn.Sequential(
                conv1,
                relu1,
                maxpool1,
                conv2,
                relu2,
                maxpool2
        
        )
        
        # full connect input dimension
        h = int((int((height - conv1.kernel_size[0] + 1) / maxpool1.kernel_size[0]) - conv2.kernel_size[0] + 1) / maxpool2.kernel_size[0])
        w = int((int((width - conv1.kernel_size[1] + 1) / maxpool1.kernel_size[1]) - conv2.kernel_size[1] + 1) / maxpool2.kernel_size[1])
        fc_in_dimen = channel_out2 * (h * w)
        
        self.fc = nn.Sequential(
                nn.Linear(fc_in_dimen, 120),
                nn.Linear(120, 84),
                nn.Linear(84, out_dimen)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.layers.forward(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    
    ##############################################
    # prepare data
    data_dir = "./data"
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())
    
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    ##############################################
    
    num_epoches = 10
    use_gpu = False

    model = CNN([1, 28, 28], 10)
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    if use_gpu:
        model.cuda()
    for epoch in range(num_epoches):
        tic()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 1):
            # [n,1,28,28], [1000]
            img, label = data  # type: torch.Tensor,torch.Tensor
            
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
        ################################
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                img, label = data
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
                100.0 * correct / total))
