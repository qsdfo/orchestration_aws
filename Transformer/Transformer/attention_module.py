import torch
import torch.nn as nn

from Transformer.data_processor import DataProcessor
from Transformer.layers import get_non_pad_mask, get_subsequent_mask, get_attn_key_pad_mask, \
    RelativeMultiHeadAttention, PositionwiseFeedForward
from Transformer.mixup import mixing

from Transformer.helpers import cuda_variable


class AttentionModule(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, data_processor: DataProcessor,
            enc_dec_conditioning,
            dropout,
            input_dropout,
            double_conditioning,
            conditioning,
            conditioner,
            shift,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.data_processor = data_processor
        self.shift = shift
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.enc_dec_conditioning = enc_dec_conditioning

        layer_stack = [
            AttentionLayer(d_model, d_inner, n_head, d_k, d_v,
                           data_processor=data_processor,
                           dropout=dropout,
                           conditioning=conditioning,
                           double_conditioning=double_conditioning,
                           enc_dec_conditioning=enc_dec_conditioning,
                           flip_masks=self.data_processor.flip_masks,
                           use_masks=self.data_processor.use_masks,
                           shift=shift)
            for _ in range(n_layers - 1)]

        self.conditioner = conditioner

        if conditioner:
            #  Write dimension for the last layer of the encoder
            last_layer_encoder = d_k * n_head * n_layers
        else:
            last_layer_encoder = None

        layer_stack.append(AttentionLayer(d_model, d_inner, n_head, d_k, d_v,
                                          data_processor=data_processor,
                                          dropout=dropout,
                                          conditioning=conditioning,
                                          double_conditioning=double_conditioning,
                                          enc_dec_conditioning=enc_dec_conditioning,
                                          flip_masks=self.data_processor.flip_masks,
                                          use_masks=self.data_processor.use_masks,
                                          shift=shift,
                                          last_layer_encoder=last_layer_encoder))

        self.layer_stack = nn.ModuleList(layer_stack)

        return

    def forward(self, x, cpc, enc_outputs, enc_enc_outputs, return_attns, embed,
                mixup_layers, mixup_lambdas):
        """
        :param enc_outputs:
        :param x: any
        :param return_attns:
        :param embed: if True, x  will be embedded
        :return:
        """

        dec_slf_attn_list, dec_enc_attn_list = [], []

        #  Todo: uise enc_enc_outputs

        ###############################################################
        # only for the sizes
        if embed:
            tgt_seq = self.data_processor.embed(x=x)
        else:
            #
            tgt_seq = x
        # must truncate if the sequence is to be shifted
        if self.shift:
            tgt_seq = tgt_seq[:, :-1, 0]
            # if enc_output is not None:
            #     enc_output = enc_output[:, :tgt_seq.size(1), :]
        else:
            tgt_seq = tgt_seq[:, :, 0]
        if mixup_lambdas is not None:
            batch_size = tgt_seq.size()[0]
            tgt_seq = tgt_seq[:batch_size // 2]
        ###############################################################"

        ###############################################################"
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)

        if self.data_processor.use_masks:
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            # slf_attn_mask = torch.zeros_like(slf_attn_mask_subseq)
            slf_attn_mask = None
        ###############################################################

        ###############################################################
        # -- Prepare inputs
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
        if self.shift:
            x_embedded = x_embedded[:, :-1, :]

        # Mixup
        if mixup_lambdas is not None:
            x_embedded = mixing(data=x_embedded, layer_ind=None, mixup_lambdas=mixup_lambdas, mixup_layers=None)
            x_embedded = x_embedded[:batch_size]

        if cpc is None:
            net_input = torch.cat([
                x_embedded,
                position[:, :truncated_sequence_length]],
                2)
        else:
            net_input = torch.cat([
                x_embedded,
                position[:, :truncated_sequence_length],
                cpc[:, :truncated_sequence_length]],
                2)

        #  Input dropout
        net_input = self.input_dropout(net_input)
        dec_output = net_input

        # In case of hierarchical attention, encoder_outputs is a list containing one encoding per layer
        # Need to convert to a list in other cases to fit with the hierarchical case
        # if type(enc_outputs) != list:
        #     enc_outputs = [enc_outputs] * len(self.layer_stack)
        ###############################################################

        ###############################################################
        # -- Forward
        dec_outputs = []

        for layer_ind, dec_layer in enumerate(self.layer_stack):

            if enc_outputs is not None:
                enc_output = enc_outputs[:, :, :, layer_ind]
            else:
                enc_output = None

            if enc_enc_outputs is not None:
                enc_enc_output = enc_enc_outputs[:, :, :, layer_ind]
            else:
                enc_enc_output = None

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output=enc_output,
                enc_enc_output=enc_enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            dec_outputs.append(dec_output)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if self.shift:
            # Append a zeros at the beginning of the shifted sequence, ignored in X-ent
            batch_size, length_minus_one, num_features = dec_output.size()
            dec_outputs[-1] = torch.cat([
                cuda_variable(torch.zeros(batch_size, 1, num_features)),
                dec_output
            ], 1)

        dec_return = dec_outputs[-1]

        if self.conditioner:
            if self.enc_dec_conditioning == 'single':
                dec_return = dec_return.unsqueeze(-1).repeat(1, 1, 1, self.n_layers)
            elif self.enc_dec_conditioning == 'split':
                dec_return = dec_return.view(batch_size, truncated_sequence_length, -1, self.n_layers)

        if return_attns:
            return dec_return, dec_slf_attn_list, dec_enc_attn_list

        return dec_return, None, None


class AttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v,
                 data_processor: DataProcessor,
                 dropout,
                 conditioning,
                 double_conditioning,
                 enc_dec_conditioning,
                 use_masks=True,
                 shift=True,
                 flip_masks=False,
                 last_layer_encoder=None
                 ):
        assert not (flip_masks and conditioning)
        super(AttentionLayer, self).__init__()
        self.conditioning = conditioning
        self.double_conditioning = double_conditioning
        self.flip_masks = flip_masks
        if self.conditioning:
            self.enc_attn = RelativeMultiHeadAttention(n_head,
                                                       d_model,
                                                       d_k,
                                                       d_v,
                                                       data_processor=data_processor,
                                                       dropout=dropout,
                                                       use_masks=use_masks,
                                                       shift=shift,
                                                       enc_dec_attention='cond',
                                                       )
        else:
            self.enc_attn = None

        if self.double_conditioning:
            self.enc_enc_attn = RelativeMultiHeadAttention(n_head,
                                                           d_model,
                                                           d_k,
                                                           d_v,
                                                           data_processor=data_processor,
                                                           dropout=dropout,
                                                           use_masks=use_masks,
                                                           shift=shift,
                                                           enc_dec_attention='double_cond',
                                                           )
        else:
            self.enc_enc_attn = None
        self.slf_attn = RelativeMultiHeadAttention(n_head,
                                                   d_model,
                                                   d_k,
                                                   d_v,
                                                   data_processor=data_processor,
                                                   dropout=dropout,
                                                   use_masks=use_masks,
                                                   shift=shift,
                                                   enc_dec_attention=None
                                                   )

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner,
                                               dropout=dropout,
                                               last_layer_encoder=last_layer_encoder,
                                               enc_dec_conditioning=enc_dec_conditioning)
        return

    def forward(self, dec_input, enc_output, enc_enc_output, non_pad_mask=None, slf_attn_mask=None):
        # No masking over encoder conditioning
        # batch_x_head = dec_input.shape[0]
        # dec_seq_len = dec_input.shape[1]
        # enc_seq_len = enc_output.shape[1]
        # dec_enc_attn_mask = torch.zeros_like(slf_attn_mask)

        if self.flip_masks:
            slf_attn_mask = slf_attn_mask\
                .type(torch.uint8)\
                .flip(1).flip(2)\
                .type(torch.bool)

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

        if self.double_conditioning:
            dec_output, dec_enc_enc_attn = self.enc_enc_attn(
                dec_output, enc_enc_output, enc_enc_output,
                mask=None,
                flipped_masks=False)
        else:
            dec_enc_attn = None

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
