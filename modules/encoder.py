import torch
import torch.nn as nn

from model_utils import clones
from model_utils import LayerNorm
from model_utils import SublayerConnection

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        '''
        x.shape = ()
        mask.shape = ()
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, lambda x: self.feed_forward)