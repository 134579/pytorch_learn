import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import Parameter


class LinearRegression(nn.Module):

    def __init__(self, d_in, d_out):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear.forward(x)
        # out = self.relu.forward(x1)
        out = x1
        return out


class LinearRegressionImpl(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = Parameter(torch.randn([in_dim, out_dim]))
        self.b = Parameter(torch.randn([1, out_dim]))

    def forward(self, x):
        return x @ self.w + self.b


if __name__ == '__main__':
    x_train = torch.Tensor(np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                                     [9.779], [6.182], [7.59], [2.167], [7.042],
                                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32))

    y_train = torch.Tensor(np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                                     [3.366], [2.596], [2.53], [1.221], [2.827],
                                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32))

    # init
    model = LinearRegression(1, 1)
    L1 = nn.MSELoss()
    L2 = nn.Softshrink(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # train

    num_epochs = 1000
    for epoch in range(num_epochs):
        # inputs = Variable(x_train)
        # target = Variable(y_train)
        inputs = x_train
        target = y_train

        # forward
        out = model(inputs)  # 前向传播
        loss = L1(out, target)  # 计算loss
        l1_reg = Variable(torch.FloatTensor(1), requires_grad=True)

        for param in model.parameters():
            l1_reg = l1_reg + param.norm(1)
        loss = loss + 0.1 * l1_reg

        # backward
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if (epoch) % 20 == 0:
            print(f'Epoch[{epoch}/{num_epochs}], loss: {loss.data}')

    # predict
    model.eval()
    x_test = torch.arange(0, 10.0, 0.1).view(-1, 1)  # type: torch.Tensor
    y_test = model(x_test)  # type: torch.Tensor

    plt.figure()
    plt.plot(x_test.data.numpy(), y_test.data.numpy(), 'r')
    plt.scatter(x_train.data.numpy(), y_train.data.numpy())
    plt.show()
