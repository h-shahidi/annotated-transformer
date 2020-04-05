import time
from copy import deepcopy as copy
import torch.nn as nn

from modules.encoder import Encoder, EncoderLayer
from modules.decoder import Decoder, DecoderLayer
from modules.encoder_decoder import EncoderDecoder
from modules.attention import MultiHeadedAttention
from modules.generator import Generator
from utils.feed_forward import PositionwiseFeedForward
from utils.positional_encoding import PositionalEncoding
from utils.embedding import Embedding


def make_model(src_vocab, 
               tgt_vocab, 
               N=6, 
               d_model=512, 
               d_ff=2048, 
               h=8, 
               dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pe = PositionalEncoding(d_model, dropout)

    encoder_layer = EncoderLayer(d_model, copy(attn), copy(ff), dropout)
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(d_model, copy(attn), copy(attn), copy(ff), dropout)
    decoder = Decoder(decoder_layer, N)

    src_embed = nn.Sequential(Embedding(src_vocab, d_model), copy(pe))
    tgt_embed = nn.Sequential(Embedding(tgt_vocab, d_model), copy(pe))

    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(data_iter, model, loss_func):
    start_time = time.time()
    total_loss = 0
    total_tokens = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_func(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start_time
            print("Epoch Step: %d Loss %f Tokens per sec %f" % 
                (i, loss / batch.ntokens, tokens / elapsed))
            tokens = 0
            start_time = time.time()
    return total_loss / total_tokens
            

if __name__ == "__main__":
    make_model(100, 100)