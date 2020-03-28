import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utils import clones

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # shape = (nbatches, nheads, query_length, key_length)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1) # shape = (nbatches, nheads, query_length, key_length)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # shape of first element = (nbatches, nheads, query_length, d_model)
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h # We assume d_v always equals d_k
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        '''
        query.shape = (nbatches, query_length, d_model)
        key.shape = (nbatches, key_length, d_model)
        value.shape = (nbatches, value_legnth, d_model), value_legnth = key_length
        mask.shape = (nbatches, query_length)
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) # shape = (nbatches, query_length, d_model)