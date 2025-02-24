# from functions import calculate_attention
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from functions import make_pad_mask, make_subsequent_mask


class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        out = sub_layer(x)
        out = self.dropout(out)
        out = out + x
        out = self.norm(out)
        return out


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2, dr_rate=0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1  # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2  # (d_ff, d_embed)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.k_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.v_fc = copy.deepcopy(qkv_fc)  # [d_embed, d_model]
        self.out_fc = out_fc
        self.dropout = nn.Dropout(p=dr_rate)

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: [n_batch, h, seq_len, d_k]
        # mask: [n_batch, 1, seq_len, seq_len]
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q * K^T, [n_batch, h, seq_len, seq_len]
        attention_score = attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)  # should this be -2? [n_batch, h, seq_len, seq_len]
        attention_prob = self.dropout(attention_prob)
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
    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, src, src_mask):
        out = self.residual1(src, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)

        return out


class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(n_layer)])
        self.norm = norm

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = self.residual1(tgt, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        return out


class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(n_layer)])
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)

        return out


class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x)*math.sqrt(self.d_embed)
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed, dr_rate=0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x):
        out = self.embedding(x)
        out = self.dropout(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cuda")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -math.log(10000.0) / d_embed)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def make_src_mask(self, src):
        pad_mask = make_pad_mask(src, src)
        return pad_mask

    def make_tgt_mask(self, tgt):
        pad_mask = make_pad_mask(tgt, tgt)
        seq_mask = make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = make_pad_mask(tgt, src)
        return pad_mask

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)

        return out, decoder_out
