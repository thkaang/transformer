import torch
import torch.nn.functional as F
import math
import numpy as np


def calculate_attention(query, key, value, mask):
    # query, key, value: [n_batch, seq_len, d_k]
    # mask: [n_batch, seq_len, seq_len]
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1))    # Q * K^T, [n_batch, seq_len, seq_len]
    attention_score = attention_score / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1)  # isn't it -2? [n_batch, seq_len, seq_len]
    out = torch.matmul(attention_prob, value)   # [n_batch, seq_len, d_k]

    return out


def make_pad_mask(query, key, pad_idx=1):
    # query: [n_batch, query_seq_len]
    # key: [n_batch, key_seq_len]
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)    # [n_batch, 1, 1, key_seq_len]
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)      # [n_batch, 1, query_seq_len, key_seq_len]

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)    # [n_batch, 1, query_seq_len, 1]
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)        # [n_batch, 1, query_seq_len, key_seq_len]

    # should be checked how this works
    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask


def make_subsequent_mask(query, key):
    # query: [n_batch, query_seq_len]
    # key: [n_batch, key_seq_len]
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)

    return mask
