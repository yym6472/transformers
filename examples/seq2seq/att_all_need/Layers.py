''' Define the Layers '''
import torch.nn as nn
import torch
from att_all_need.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiHeadAttention_extra_mask


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn1 = MultiHeadAttention_extra_mask(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn2 = MultiHeadAttention_extra_mask(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn3 = MultiHeadAttention_extra_mask(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask, extra_mask1, extra_mask2, extra_mask3):
        # enc_output, enc_slf_attn, enc_hidden_states = self.slf_attn1(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask, extra_mask=extra_mask1,ratio=0)
        enc_output1, enc_slf_attn1, enc_hidden_states1 = self.slf_attn2(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, extra_mask=extra_mask2, ratio=0.2)
        enc_output2, enc_slf_attn2, enc_hidden_states2 = self.slf_attn3(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, extra_mask=extra_mask3, ratio=0.2)
        enc_output, enc_slf_attn, enc_hidden_states = self.slf_attn1(
            enc_output2, enc_output1, enc_output1, mask=slf_attn_mask, extra_mask=extra_mask1,ratio=0.2)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn, enc_hidden_states


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask, dec_enc_attn_mask):
        dec_output, dec_slf_attn, dec_hidden_states = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn, enc_dec_contexts = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn, dec_hidden_states, enc_dec_contexts
