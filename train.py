import copy
import torch
from model import *


def build_default_model(src_vocab_size,
                        tgt_vocab_size,
                        device=torch.device("cuda"),
                        max_len=256,
                        d_embed=512,
                        n_layer=6,
                        d_model=512,
                        h=8,
                        d_ff=2048,
                        dr_rate=0.1,
                        norm_eps=1e-5):
    src_token_embed = TokenEmbedding(d_embed=d_embed,
                                     vocab_size=src_vocab_size)

    tgt_token_embed = TokenEmbedding(d_embed=d_embed,
                                     vocab_size=tgt_vocab_size)

    pos_embed = PositionalEncoding(d_embed=d_embed,
                                   max_len=max_len,
                                   device=device)

    src_embed = TransformerEmbedding(token_embed=src_token_embed,
                                     pos_embed=copy.deepcopy(pos_embed))

    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed,
                                     pos_embed=copy.deepcopy(pos_embed))

    attention = MultiHeadAttentionLayer(d_model=d_model,
                                        h=h,
                                        qkv_fc=nn.Linear(d_embed, d_model),
                                        out_fc=nn.Linear(d_model, d_embed))

    position_ff = PositionWiseFeedForwardLayer(fc1=nn.Linear(d_embed, d_ff),
                                               fc2=nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(self_attention=copy.deepcopy(attention),
                                 position_ff=copy.deepcopy(position_ff))

    decoder_block = DecoderBlock(self_attention=copy.deepcopy(attention),
                                 cross_attention=copy.deepcopy(attention),
                                 position_ff=copy.deepcopy(position_ff))

    encoder = Encoder(encoder_block=encoder_block,
                      n_layer=n_layer)

    decoder = Decoder(decoder_block=decoder_block,
                      n_layer=n_layer)

    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(src_embed=src_embed,
                        tgt_embed=tgt_embed,
                        encoder=encoder,
                        decoder=decoder,
                        generator=generator)

    model.device = device

    return model

