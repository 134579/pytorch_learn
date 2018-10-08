import torch

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from misc import tic, toc


class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        hidden_dim = 64
        self.in_dim, self.out_dim, self.num_layers = in_dim, out_dim, num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
    
    def forward1(self, x):
        # only one channel
        x = x[:, 0, :, :]  # type: torch.Tensor
        num_directions = 1
        # (batch, num_layers * num_directions, , hidden_size)
        batch_size, seq_len, in_dim = x.size()
        
        h = torch.zeros((1, self.num_layers * num_directions, out_dim), dtype=torch.double).cuda()
        c = torch.zeros((1, self.num_layers * num_directions, out_dim), dtype=torch.double).cuda()
        
        outputs = []
        for i, input_t in enumerate(x.chunk(batch_size, dim=0)):
            out, (h, c) = self.lstm.forward(input_t, (h, c))  # (1,seq_len, out_dim)
            h = h.detach()
            c = c.detach()
            outputs.append(out)
        
        class_ = torch.stack(outputs).squeeze(dim=1)  # (1000, 28, 10)
        
        return class_[:, -1, :]
    
    def forward(self, x):
        # only one channel
        x = x[:, 0, :, :]  # type: torch.Tensor
        num_directions = 1
        out, _ = self.lstm(x)  # (batch_size,seq_len, num_directions * hidden_size)
        lstm_out = out[:, -1, :]
        return self.linear(lstm_out)


class PureLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, num_layers=num_layers, batch_first=True)
    
    def forward(self, x):
        # only one channel
        x = x[:, 0, :, :]  # type: torch.Tensor
        num_directions = 1
        out, _ = self.lstm(x)  # (batch_size,seq_len, num_directions * hidden_size)
        lstm_out = out[:, -1, :]
        return lstm_out


if __name__ == '__main__':
    in_dim = 28
    out_dim = 10
    num_layers = 3
    batch_size = 100
    learning_rate = 1e-2
    epochs = 100
    
    use_gpu = True
    lstm = LSTM(in_dim, out_dim, num_layers)
    if use_gpu:
        lstm = lstm.cuda()
    
    data_dir = "./data"
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # (batch_size, channel, height, weight)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)
    # optimizer = optim.Adamax(lstm.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        tic()
        accumulative_loss = 0.0
        tp = 0.0  # true positive
        sample_size = 0
        for i, data in enumerate(train_loader, 1):
            lstm.train()
            img, label = data  # type: torch.Tensor,torch.Tensor
            
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            assert img.size(0) == batch_size and label.size(0) == batch_size
            sample_size += img.size(0)
            optimizer.zero_grad()
            out = lstm.forward(img)
            
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            accumulative_loss += loss.item()
            
            # calculate true positive and accumulative loss
            _, predicted = torch.max(out, 1)
            tp += (predicted == label).sum().item()
            
            if i % 100 == 0:  # print every 2000 mini-batches
                # accuracy
                accuracy = tp / sample_size
                print(f'[{sample_size:5d},{epoch + 1:d}] loss: {accumulative_loss:.3f}, acc: {accuracy :.3f}')
                running_loss = 0.0
        toc()
        
        correct = 0
        total = 0
        with torch.no_grad():
            lstm.eval()
            for i, data in enumerate(test_loader):
                img, label = data
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                
                out = lstm.forward(img)
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100.0 * correct / total:f} %')
