import torch
import torch.nn as nn

from utils.utils import clones
from utils.layer_norm import LayerNorm
from utils.sublayer_connection import SublayerConnection

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
        x.shape = (nbatches, src_length, d_model)
        mask.shape = (nbatches, 1, src_length)
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)