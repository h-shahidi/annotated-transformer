import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None
        
    def forward(self, x, target):
        '''
        x is the prediction with shape = (n_tokens, vocab_size)
        target.shape = (n_tokens)
        '''
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2)) # subtract one for padding token and one for the true token.
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == "__main__":
    criterion = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                [0, 0.2, 0.7, 0.1, 0], 
                                [0, 0.2, 0.7, 0.1, 0]])
    v = criterion(Variable(predict.log()), 
             Variable(torch.LongTensor([2, 1, 0])))
    plt.figure()
    plt.imshow(criterion.true_dist)
    plt.savefig("label_smoothing_true_dist.png")

    criterion = LabelSmoothing(5, 0, 0.1)
    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
        return criterion(Variable(predict.log()),
            Variable(torch.LongTensor([1]))).data
    plt.figure()
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])    
    plt.savefig("label_smoothing_loss.png")