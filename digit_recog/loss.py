from pprint import pprint

import torch
import torch.nn as nn
from digit_recog.Net import Net

if __name__ == '__main__':
    net = Net()
    # run net
    input = torch.randn([1, 1, 32, 32])
    target = torch.randn([1, 10])
    out = net.forward(input)

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU