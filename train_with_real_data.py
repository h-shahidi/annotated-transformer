from utils.data_loading import load_data
from utils.train import make_model
from utils.label_smoothing import LabelSmoothing

devices = [5, 6]


if __name__ == "__main__":
    train, val, test, SRC, TGT = load_data()
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(vocab_size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()