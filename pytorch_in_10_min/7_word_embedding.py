import torch
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':
    word_to_ix = {'hello': 0, 'world': 1}
    embeds = nn.Embedding(2, 5)
    hello_idx = torch.LongTensor([word_to_ix['hello']])
    # hello_idx = Variable(hello_idx)
    hello_embed = embeds(hello_idx)
    print(hello_embed)
