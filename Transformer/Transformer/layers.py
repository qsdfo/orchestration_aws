import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F

from Transformer import constants
from Transformer.helpers import cuda_variable


class RelativeMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v,
                 data_processor,
                 dropout,
                 use_masks,
                 shift,
                 enc_dec_attention):
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

        # if self.enc_dec_attention:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(q[0, :, 0].detach().cpu().numpy())
        #     plt.savefig('plots/Q')
        #     plt.imshow(k[0, :, 0].detach().cpu().numpy())
        #     plt.savefig('plots/K')
        #     plt.imshow(torch.mm(q[0, :, 0], k[0, :, 0].t()).detach().cpu().numpy())
        #     plt.savefig('plots/QKt')

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        q2 = q2.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention.forward(q, k, v, self.r_module, q2, mask=mask,
                                              flipped_masks=flipped_masks)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, n_head * d_v)
        # Mix heads
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
        #  Todo Careful with shift for absolute (block) positioning !!!!
        # compute attention
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # compute relative attention
        # dedicated q for relative attention or not
        if q2 is None:
            q2 = q
        rel_attn = cuda_variable(r_module.forward(q=q2, flipped_masks=flipped_masks))
        rel_attn = rel_attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
            rel_attn = rel_attn.masked_fill(mask, -np.inf)

        out_attn = attn + rel_attn
        out_attn = self.softmax(out_attn)

        #  Dropout here ?
        # out_attn_dropout = self.dropout(out_attn)
        # output = torch.bmm(out_attn_dropout, v)
        output = torch.bmm(out_attn, v)

        return output, out_attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout, last_layer_encoder, enc_dec_conditioning):
        super().__init__()
        #  Alternative:
        #  One layer, d_hid being the dimension for conditioning the decoder
        if (last_layer_encoder is None) or (enc_dec_conditioning == 'single'):
            d_out = d_in
            self.residual_repeat = None
        else:
            d_out = last_layer_encoder
            self.residual_repeat = d_out // d_in  #  Actually equals to num_layers
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_out, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.residual_repeat:
            #  Concatenate residual
            residual = residual.repeat(1, 1, self.residual_repeat)
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)
        return output


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD).type(torch.uint8)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

#
# class Left(nn.Module):
#     ''' A decoder model with self attention mechanism. '''
#
#     def __init__(
#             self, get_len_max_seq,
#             n_layers, n_head, d_k, d_v,
#             d_model, d_inner, d_ticks, note_embeddings, dropout=0.1,
#             flip=False):
#
#         super().__init__()
#
#         self.note_embeddings = note_embeddings
#
#         self.layer_stack = nn.ModuleList([
#             DecoderLayer(d_model, d_inner, n_head, d_k, d_v, get_len_max_seq, dropout=dropout)
#             for _ in range(n_layers)])
#
#         self.intrument_labels = cuda_variable(
#             torch.Tensor([(i % 16) for i in range(
#                 get_len_max_seq)]).long())
#
#         self.instrument_embedding = nn.Embedding(16, d_ticks)
#         self.flip = flip
#
#     def forward(self, chorale, return_attns=False, shift=True):
#         """
#         :param chorale: (:, num_voices, length)
#         :param return_attns:
#         :return:
#         """
#         dec_slf_attn_list, dec_enc_attn_list = [], []
#
#         # todo not optimal
#         # only for the sizes
#         if self.flip:
#             tgt_seq = self.flatten_chorale(chorale)[:, 1:]
#         else:
#             tgt_seq = self.flatten_chorale(chorale)[:, :-1]
#
#         # -- Prepare masks
#         non_pad_mask = get_non_pad_mask(tgt_seq)
#
#         slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
#         slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
#         slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
#
#         # -- Forward
#         # TODO how to compute position?
#         # TODO sum or concat?
#
#         batch_size, num_voices, chorale_length = chorale.size()
#         # higher validation loss with position?!
#         # dec_output = self.embed_chorale(chorale)[:, :-1]
#
#         position = self.instrument_embedding(self.intrument_labels.unsqueeze(0).repeat(
#             batch_size, 1))
#
#         if self.flip:
#             dec_output = torch.cat([
#                 self.embed_chorale(chorale)[:, 1:],
#                 position[:, :chorale_length * num_voices - 1]],
#                 2)
#         else:
#             dec_output = torch.cat([
#                 self.embed_chorale(chorale)[:, :-1],
#                 position[:, :chorale_length * num_voices - 1]],
#                 2)
#
#         # flip masks if necessary
#         if self.flip:
#             slf_attn_mask = slf_attn_mask.flip(1)
#
#         for dec_layer in self.layer_stack:
#             dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
#                 dec_output,
#                 non_pad_mask=non_pad_mask,
#                 slf_attn_mask=slf_attn_mask)
#
#             if return_attns:
#                 dec_slf_attn_list += [dec_slf_attn]
#                 dec_enc_attn_list += [dec_enc_attn]
#
#         if shift:
#             # add dummy input on first pos or last pos depending on flip
#             if self.flip:
#                 batch_size, length_minus_one, num_features = dec_output.size()
#                 dec_output = torch.cat([
#                     dec_output,
#                     cuda_variable(torch.zeros(batch_size, 1, num_features))
#                 ], 1)
#             else:
#                 batch_size, length_minus_one, num_features = dec_output.size()
#                 dec_output = torch.cat([
#                     cuda_variable(torch.zeros(batch_size, 1, num_features)),
#                     dec_output
#                 ], 1)
#         else:
#             pass
#         if return_attns:
#             return dec_output, dec_slf_attn_list, dec_enc_attn_list
#
#         return dec_output,
#
