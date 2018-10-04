import torch
import torch.nn as nn
from torch.nn import Parameter


class RNN(torch.nn.Module):
    def __init__(self, in_dimen, out_dimen):
        super().__init__()
        wih = Parameter(torch.Tensor())
        nn.RNNCell
        nn.RNN

    def forward(self, x: torch.Tensor):
        pass
