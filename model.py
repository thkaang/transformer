# from functions import calculate_attention
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = sub_layer(x)
        out = out + x
        return out


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1  # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2  # (d_ff, d_embed)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.k_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.v_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.out_fc = out_fc

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: [n_batch, h, seq_len, d_k]
        # mask: [n_batch, 1, seq_len, seq_len]
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q * K^T, [n_batch, h, seq_len, seq_len]
        attention_score = attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)  # should this be -2? [n_batch, h, seq_len, seq_len]
        out = torch.matmul(attention_prob, value)  # [n_batch, h, seq_len, d_k]

        return out

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: [n_batch, seq_len, d_embed]
        # mask: [n_batch, seq_len, seq_len]
        # return value: [n_batch, seq_len, d_embed]
        n_batch = query.size(0)

        def transform(x, fc):   # [n_batch, seq_len, d_embed]
            out = fc(x)     # [n_batch, seq_len, d_model]
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)   # [n_batch, seq_len, h, d_k]
            out = out.transpose(1, 2)   # [n_batch, h, seq_len, d_k]
            return out

        query = transform(query, self.q_fc)     # [n_batch, h, seq_len, d_K]
        key = transform(key, self.k_fc)         # [n_batch, h, seq_len, d_K]
        value = transform(value, self.v_fc)     # [n_batch, h, seq_len, d_K]

        out = self.calculate_attention(query, key, value, mask)  # [n_batch, h, seq_len, d_k]
        out = out.transpose(1, 2)   # [n_batch, seq_len, h, d_k]
        out = out.contiguous().view(n_batch, -1, self.d_model)  # [n_batch, seq_len, d_model]
        out = self.out_fc(out)  # [n_batch, seq_len, d_embed]
        return out


class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]

    def forward(self, src, src_mask):
        out = self.residuals[0](src, lambda src: self.self_attention(query=src, key=src, value=src, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)

        return out


class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)

        return out


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_pad_mask(self, query, key, pad_idx=1):
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

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def forward(self, src, tgt, src_mask):
        encoder_out = self.encoder(src, src_mask)
        y = self.decoder(tgt, encoder_out)

        return y