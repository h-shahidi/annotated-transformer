import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class NoamOpt():
    def __init__(self, model_size, factor, warmup, optimizer):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._rate = 0
        self._step = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
               self.model_size ** (-0.5) * \
               min(step ** (-0.5), step * self.warmup ** (-1.5))

if __name__ == "__main__":
    opt = NoamOpt(512, 1, 4000, None)
    plt.plot(np.arange(1, 20000), [opt.rate(i) for i in range(1, 20000)])
    plt.savefig("learning_rate.png")