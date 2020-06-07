import torch
import torch.nn as nn
from torch.autograd import Variable


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm 


class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        self.generator = nn.parallel.replicate(generator, devices=devices)
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __class__(self, out, targets, norm):
        total = 0.0
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        for i in range(0, out_scatter[0].size(1), self.chunk_size):
            # Predict distributions
            out_column = []
            for o in out_scatter:
                out_column.append([Variable(o[:, i:i+self.chunk_size].data, 
                                            requires_grad=self.opt is not None)])
            gen = nn.parallel.parallel_apply(self.generator, out_column)

            # Compute loss.
            y = []
            for g,t in zip(gen, targets):
                y.append((g.contiguous().view(-1, g.size(-1)), 
                          t[:, i:i+self.chunk_size].contiguous().view(-1)))
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0] / norm
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        
        return total * norm 
