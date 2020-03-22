import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        '''
        x.shape = ()
        '''
        mean = x.mean(-1, keepdim=True) # shape = ()
        std = x.std(-1, keepdim=True) # shape = ()
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


if __name__ == "__main__":
    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20)[0])
    plt.savefig("foo.png")