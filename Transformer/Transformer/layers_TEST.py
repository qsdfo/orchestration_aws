import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F

from Transformer import constants


class RelativeMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, len_max_seq,
                 data_processor,
                 dropout=0.1,
                 use_masks=True,
                 shift=None,
                 enc_dec_attention=False):
        super().__init__()
        self.data_processor = data_processor
        self.enc_dec_attention = enc_dec_attention

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # todo linear embeddings/ more complex for Q, K and V?
        # TODO no bias?!
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # todo ADD PARAMETER SELECTION FOR SECOND Q
        self.w_q2s = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.normal_(self.w_q2s.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.r_module = self.data_processor.get_relative_attention_module(
            embedding_dim=d_k,
            n_head=n_head,
            max_len_seq=len_max_seq,
            use_masks=use_masks,
            shift=shift,
            enc_dec_attention=enc_dec_attention
        )

        self.attention = RelativeScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                           attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v,
                mask=None,
                flipped_masks=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # TODO adapt signature to add q2 (useful for encoders)
        #  for decoder here q = k = v = x
        residual = q

        q2 = self.w_q2s(q).view(sz_b, len_q, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        q2 = q2.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention.forward(q, k, v, self.r_module, q2, mask=mask,
                                              flipped_masks=flipped_masks)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_h*dv)
        # todo
        #  Here is the part about not concatenating the heads (fc does the mixing between heads)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class RelativeScaledDotProductAttention(nn.Module):
    ''' Relative Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, r_module, q2=None, mask=None, flipped_masks=False):
        # Todo Careful with shift for absolute (block) positioning !!!!
        # compute attention
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # compute relative attention
        # dedicated q for relative attention or not
        if q2 is None:
            q2 = q
        rel_attn = r_module.forward(q=q2, flipped_masks=flipped_masks)
        rel_attn = rel_attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
            rel_attn = rel_attn.masked_fill(mask, -np.inf)

        out_attn = attn + rel_attn
        out_attn = self.softmax(out_attn)
        out_attn_dropout = self.dropout(out_attn)
        output = torch.bmm(out_attn_dropout, v)
        return output, out_attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask