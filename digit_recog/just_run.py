from pprint import pprint

import torch

from digit_recog.Net import Net

if __name__ == '__main__':
    net = Net()
    print(net)
    params = list(net.parameters())
    pprint([param.size() for param in params])

    # run net
    input = torch.randn([3, 1, 32, 32])
    out = net.forward(input)
    print(out)

    net.zero_grad()
    out.backward(torch.ones(3, 10))
