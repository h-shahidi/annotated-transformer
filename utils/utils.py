import copy
import math
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


if __name__ == "__main__":
    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20)[0])
    plt.savefig("subsequent-mask.png")