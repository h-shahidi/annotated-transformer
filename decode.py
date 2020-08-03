import torch

from utils.train import make_model
from utils.data_loading import load_data
from utils.batch import MyIterator, batch_size_fn
from train_with_synthetic_data import greedy_decode

BATCH_SIZE = 6000
SAVE_PATH = "./checkpoint"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    train, val, test, SRC, TGT = load_data()

    valid_iter = MyIterator(
        val,
        batch_size=BATCH_SIZE,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=False)

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.to(device)
    model.load_state_dict(torch.load(SAVE_PATH))

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, 
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end =" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end =" ")
        print()
        break