import torch
import torch.nn as nn

from utils.data_loading import load_data
from utils.train import make_model, run_epoch
from utils.label_smoothing import LabelSmoothing
from utils.batch import MyIterator, batch_size_fn, rebatch
from utils.optimizer import NoamOpt
from utils.loss import LossCompute


N_EPOCH = 10
BATCH_SIZE = 6000
SAVE_PATH = "./checkpoints"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    train, val, test, SRC, TGT = load_data()
    pad_idx = TGT.vocab.stoi["<blank>"]

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.to(device)

    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    train_iter = MyIterator(
        train, 
        batch_size=BATCH_SIZE,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=True)
    valid_iter = MyIterator(
        val,
        batch_size=BATCH_SIZE,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=False)

    model_opt = NoamOpt(
        model.src_embed[0].d_model, 
        1, 
        2000, 
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    for epoch in range(N_EPOCH):
        model.train()
        run_epoch(
            (rebatch(pad_idx, b) for b in train_iter),
            model,
            LossCompute(
                model.generator, 
                criterion,
                model_opt
            )
        )

        model.eval()
        loss = run_epoch(
            (rebatch(pad_idx, b) for b in valid_iter),
            model,
            LossCompute(
                model.generator, 
                criterion,
                None
            )
        )

    torch.save(model.state_dict(), SAVE_PATH)