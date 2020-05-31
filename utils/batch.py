from torchtext import data

from utils.utils import subsequent_mask
from torch.autograd import Variable

class Batch():
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MyIterator(data.Iterator):
    '''
    Batching matters for speed. We want to have very evenly divided batches, 
    with absolutely minimal padding. To do this we have to hack a bit around the 
    default torchtext batching. This code patches their default batching to make 
    sure we search over enough sentences to find tight batches.
    '''
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    sorted(p, key=self.sort_key)
                    p_batch = data.batch(p, self.batch_size, self.batch_size_fn):
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                sorted(b, key=self.sort_key)
                self.batches.append(b)


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src = batch.src.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)