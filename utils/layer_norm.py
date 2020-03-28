import torch
import torch.nn as nn


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