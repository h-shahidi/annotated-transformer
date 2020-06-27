import torch
import torch.nn as nn

from utils.data_loading import load_data
from utils.train import make_model
from utils.label_smoothing import LabelSmoothing
from utils.batch import MyIterator, batch_size_fn
from utils.optimizer import NoamOpt

N_EPOCH = 10
BATCH_SIZE = 12000
devices = [5, 6]


if __name__ == "__main__":
    train, val, test, SRC, TGT = load_data()
    pad_idx = TGT.vocab.stoi["<blank>"]

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()

    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()

    train_iter = MyIterator(train, 
                            batch_size=BATCH_SIZE, 
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True)
    valid_iter = MyIterator(val,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=False)

    model_par = nn.DataParallel(model, device_ids=devices)

    NoamOpt(model.src_embed[0].d_model, 
            1, 
            2000, 
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
