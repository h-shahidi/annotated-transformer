import math
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)