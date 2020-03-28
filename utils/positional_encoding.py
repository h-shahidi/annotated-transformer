import math
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        self.pe[:, 0::2] = torch.sin(position / div_term)
        self.pe[:, 1::2] = torch.cos(position / div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    pe = PositionalEncoding(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8])
    plt.savefig("positional-encoding.png")