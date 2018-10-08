import torch
import torch.nn as nn
from torch.nn import Parameter

if __name__ == '__main__':
    in_dim = 10
    out_dim = 30
    lstm = nn.LSTM(in_dim, out_dim, batch_first=True)
    batch_count = 100
    seq_len = 14
    x = torch.randn((batch_count, seq_len, in_dim))
    out, (h_n, c_n) = lstm.forward(x)
    class_ = out[:, -1, :]
