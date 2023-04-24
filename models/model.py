import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, en_input, de_input, out_size, train_len, delead_len, pred_len,
                 factors=5, d_model=512, n_heads=8, en_layers=3, de_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = pred_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.embedding_en = DataEmbedding(en_input, d_model, embed, freq, dropout)
        self.embedding_de = DataEmbedding(de_input, d_model, embed, freq, dropout)
        # Attention
        if attn == 'prob':
            Informer_attn = ProbAttention
        else:
            Informer_attn = FullAttention

        # Encoder
        self.Informer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Informer_attn(False, factors, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(en_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(en_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.Informer_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Informer_attn(True, factors, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factors, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(de_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.lineartrans = nn.Linear(d_model, out_size, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print(x_enc.shape)
        #print(x_mark_enc.shape)

        en_output = self.embedding_en(x_enc, x_mark_enc)
        en_output, for_attns = self.Informer_encoder(en_output, attn_mask=None)
        #print(x_dec.shape)
        #print(x_mark_dec.shape)

        de_output = self.embedding_de(x_dec, x_mark_dec)
        de_output = self.Informer_decoder(de_output, en_output, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #print(de_output.shape)
        de_output = self.lineartrans(de_output)
        #print(de_output.shape)

        if self.output_attention:
            return de_output[:, -self.pred_len:, :], for_attns
        else:
            return de_output[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, en_input, de_input, out_size, train_len, delead_len, pred_len,
                 factors=5, d_model=512, n_heads=8, en_layers=[3, 2, 1], de_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = pred_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.embedding_en = DataEmbedding(en_input, d_model, embed, freq, dropout)
        self.embedding_de = DataEmbedding(de_input, d_model, embed, freq, dropout)
        # Attention
        if attn == 'prob':
            Informerstack_attn = ProbAttention
        else:
            Informerstack_attn = FullAttention  # Encoder

        en_num = list(range(len(en_layers)))  # [0,1,2,...] you can customize here
        muti_encoder = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Informerstack_attn(False, factors, attention_dropout=dropout,
                                                          output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for _ in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in en_layers
        ]
        self.Infstack_encoder = EncoderStack(muti_encoder, en_num)
        # Decoder
        self.Infstack_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Informerstack_attn(True, factors, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factors, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(de_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.lineartrans = nn.Linear(d_model, out_size, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        en_output = self.embedding_en(x_enc, x_mark_enc)
        en_output, attns = self.Infstack_encoder(en_output, attn_mask=None)

        de_output = self.embedding_de(x_dec, x_mark_dec)
        de_output = self.Infstack_decoder(de_output, en_output, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        de_output = self.lineartrans(de_output)

        if self.output_attention:
            return de_output[:, -self.pred_len:, :], attns
        else:
            return de_output[:, -self.pred_len:, :]  # [B, L, D]
