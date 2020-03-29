import torch
import torch.nn as nn

from Transformer.data_processor import DataProcessor
from Transformer.layers import get_non_pad_mask, get_subsequent_mask, get_attn_key_pad_mask, \
    RelativeMultiHeadAttention, PositionwiseFeedForward


class AttentionModule(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self, len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, data_processor: DataProcessor,
            dropout=0.1,
            conditioning=True,
            use_masks=True,
            flip_masks=False,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.flip_masks = flip_masks
        self.data_processor = data_processor
        self.use_masks = use_masks
        self.n_layers = n_layers

        # Use list of 1 head modules, and concatenate after data have been passed through all heads
        self.heads = nn.ModuleList([
            nn.ModuleList([
                AttentionLayer(d_model, d_inner, 1, d_k, d_v, len_max_seq,
                               data_processor=data_processor,
                               dropout=dropout,
                               conditioning=conditioning,
                               flip_masks=flip_masks,
                               use_masks=use_masks)
                for _ in range(n_layers)])
            for _ in range(n_head)])

        return

    def forward(self, x, enc_outputs, return_attns=False, return_all_layers=False, shift=True, embed=True):
        """
        :param enc_outputs:
        :param return_all_layers:
        :param x: any
        :param return_attns:
        :param shift:
        :param embed: if True, x  will be embedded
        :return:
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # todo not optimal
        # only for the sizes
        if embed:
            tgt_seq = self.data_processor.embed(x=x)
        else:
            #
            tgt_seq = x

        # must truncate if the sequence is to be shifted
        if shift:
            tgt_seq = tgt_seq[:, :-1, 0]
            # if enc_output is not None:
            #     enc_output = enc_output[:, :tgt_seq.size(1), :]
        else:
            tgt_seq = tgt_seq[:, :, 0]

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)

        if self.use_masks:
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = torch.zeros_like(slf_attn_mask_subseq)

        # -- Forward
        batch_size, truncated_sequence_length = tgt_seq.size()

        # Get local positions embeddings
        position = self.data_processor.local_position(batch_size=batch_size,
                                                      sequence_length=truncated_sequence_length)
        if embed:
            x_embedded = self.data_processor.embed(x)
        else:
            # otherwise, x should be already embedded
            x_embedded = x

        # must truncate if the sequence is to be shifted
        if shift:
            x_embedded = x_embedded[:, :-1, :]

        dec_output = torch.cat([
            x_embedded,
            position[:, :truncated_sequence_length]],
            2)

        # In case of hierarchical attention, encoder_outputs is a list containing one encoding per layer
        # Need to convert to a list in other cases to fit with the hierarchical case
        if type(enc_outputs) != list:
            enc_outputs = [enc_outputs] * self.n_layers

        dec_outputs = []

        # Todo LOOP OVER THE HEADS
        for head in self.heads:
            for dec_layer, enc_output in zip(self.layer_stack, enc_outputs):
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output,
                    enc_output=enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)

                dec_outputs.append(dec_output)

                if return_attns:
                    dec_slf_attn_list += [dec_slf_attn]
                    dec_enc_attn_list += [dec_enc_attn]

        if shift:
            # Append a zeros at the beginning of the shifted sequence, ignored in X-ent
            batch_size, length_minus_one, num_features = dec_output.size()
            dec_outputs[-1] = torch.cat([
                torch.zeros(batch_size, 1, num_features).cuda(),
                dec_output
            ], 1)

        if return_all_layers:
            dec_return = dec_outputs
        else:
            dec_return = dec_outputs[-1]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_return,


class AttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, len_max_seq,
                 data_processor: DataProcessor,
                 dropout=0.1,
                 conditioning=True,
                 use_masks=True,
                 flip_masks=False
                 ):
        assert not (flip_masks and conditioning)
        super(AttentionLayer, self).__init__()
        self.conditioning = conditioning
        self.flip_masks = flip_masks
        if self.conditioning:
            self.enc_attn = RelativeMultiHeadAttention(n_head,
                                                       d_model,
                                                       d_k,
                                                       d_v,
                                                       data_processor=data_processor,
                                                       len_max_seq=len_max_seq,
                                                       dropout=dropout,
                                                       use_masks=use_masks,
                                                       enc_dec_attention=True
                                                       )
        else:
            self.enc_attn = None
        self.slf_attn = RelativeMultiHeadAttention(n_head,
                                                   d_model,
                                                   d_k,
                                                   d_v,
                                                   data_processor=data_processor,
                                                   len_max_seq=len_max_seq,
                                                   dropout=dropout,
                                                   use_masks=use_masks,
                                                   enc_dec_attention=False
                                                   )

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        # self.layer_norm = nn.LayerNorm(d_model)
        return

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None):
        # No masking over encoder conditioning
        # batch_x_head = dec_input.shape[0]
        # dec_seq_len = dec_input.shape[1]
        # enc_seq_len = enc_output.shape[1]
        # dec_enc_attn_mask = torch.zeros_like(slf_attn_mask)

        if self.flip_masks:
            slf_attn_mask = slf_attn_mask.flip(1).flip(2)

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input,
            mask=slf_attn_mask,
            flipped_masks=self.flip_masks)
        dec_output *= non_pad_mask

        if self.conditioning:
            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output,
                mask=None,
                flipped_masks=False)
        else:
            dec_enc_attn = None

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
