import random

import music21
import numpy as np
import torch
from torch import nn

from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.helpers import REST_SYMBOL, END_SYMBOL, START_SYMBOL, UNKNOWN_SYMBOL, PAD_SYMBOL, MASK_SYMBOL

from Transformer.relative_attentions import NoPositionwiseAttention, RelativeSelfAttentionModule, \
    BlockSelfAttentionModule
from Transformer.data_processor import DataProcessor
from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, quantize_velocity_pianoroll_frame, \
    new_events


class ArrangementDataProcessor(DataProcessor):
    def __init__(self, dataset: ArrangementDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 flag,
                 block_attention,
                 nade,
                 double_conditioning,
                 # instrument_presence_in_encoder
                 ):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """
        # local_position_dim = d_ticks
        super().__init__(dataset=dataset,
                         embedding_dim=embedding_dim)

        self.dataset = dataset
        self.flag = flag
        self.block_attention = block_attention
        self.flip_masks = False
        self.double_conditioning = double_conditioning

        if (flag == 'orchestra') and not nade:
            self.use_masks = True
        else:
            self.use_masks = False

        # Useful parameters
        self.num_frames_orchestra = int(
            (dataset.sequence_size + 1) / 2)  # Only past and present is used for the orchestration
        self.num_instruments = dataset.number_instruments
        #
        self.num_frames_piano = dataset.sequence_size  # For the piano, future is also used
        self.num_pitch_piano = dataset.number_pitch_piano

        self.num_notes_per_instrument_orchestra = [len(v) for k, v in self.dataset.index2midi_pitch.items()]
        self.num_notes_per_instrument_piano = [len(v) for k, v in self.dataset.value2oneHot_perPianoToken.items()]

        # Generic names
        self.max_len_sequence_decoder = self.num_instruments * self.num_frames_orchestra
        self.max_len_sequence_encoder = self.num_pitch_piano * self.num_frames_piano
        self.max_len_sequence_encodencoder = dataset.instrument_presence_dim
        if self.flag == 'orchestra':
            self.num_token_per_tick = self.num_notes_per_instrument_orchestra
            self.num_ticks_per_frame = self.num_instruments
            self.num_frames = self.num_frames_orchestra
            max_len_sequence = self.max_len_sequence_decoder
        elif self.flag == 'piano':
            self.num_token_per_tick = self.num_notes_per_instrument_piano
            self.num_ticks_per_frame = self.num_pitch_piano
            self.num_frames = self.num_frames_piano
            max_len_sequence = self.max_len_sequence_encoder
        elif self.flag == 'instruments':
            self.num_token_per_tick = [len(dataset.index2instruments_presence)] * dataset.instrument_presence_dim
            self.num_ticks_per_frame = dataset.instrument_presence_dim
            self.num_frames = 1
            max_len_sequence = self.max_len_sequence_decoder
        else:
            raise Exception('Unrecognized data processor flag')

        self.reducer_input_dim = reducer_input_dim
        self.local_position_embedding_dim = local_position_embedding_dim

        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, self.embedding_dim)
                for num_embeddings in self.num_token_per_tick
            ]
        )

        self.linear_output_notes = nn.ModuleList(
            [
                nn.Linear(reducer_input_dim, num_notes)
                for num_notes in self.num_token_per_tick
            ]
        )

        # labels and embeddings for local position
        self.instrument_labels = nn.Parameter(
            torch.Tensor([(i % self.num_ticks_per_frame) for i in range(max_len_sequence)]).long(),
            requires_grad=False)
        self.instrument_embedding = nn.Embedding(self.num_ticks_per_frame,
                                                 local_position_embedding_dim)

        self.name = 'arrangement_data_processor'
        return


    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        piano, orchestra, instruments = tensors
        # Here extract information used by the network
        if self.flag == 'orchestra':
            # Use only past and present
            orchestra_past_present = orchestra[:, :self.num_frames_orchestra]
            return orchestra_past_present.long(), None
        elif self.flag == 'piano':
            return piano.long(), None
        elif self.flag == 'instruments':
            instruments_sum = instruments.sum(dim=1, keepdim=True)
            ret = torch.where(instruments_sum > 0, torch.ones_like(instruments_sum), torch.zeros_like(instruments_sum))
            return ret.long(), None
        else:
            raise Exception('Unrecognized data processor flag')

    @staticmethod
    def prepare_target_for_loss(target):
        return target

    @staticmethod
    def prepare_mask_for_loss(mask):
        return mask

    def embed(self, x):
        """
        :param x: (batch_size, time, num_voices)
        :return: seq: (batch_size, time * num_voices, embedding_size)
        """
        shape_x = x.size()
        if len(shape_x) == 2:
            #  Deal with the case of single piano frame
            x_permute = x.permute(1, 0)
            x_embeds = [
                embedding(voice)
                for voice, embedding
                in zip(x_permute, self.note_embeddings)
            ]
            stacked_embedding = torch.stack(x_embeds)
            ret = stacked_embedding.permute(1, 0, 2)
        else:
            x_permute = x.permute(2, 0, 1)
            x_embeds = [
                embedding(voice)
                for voice, embedding
                in zip(x_permute, self.note_embeddings)
            ]
            stacked_embedding = torch.stack(x_embeds).permute(1, 2, 0, 3)
            ret = self.flatten(stacked_embedding)
        return ret

    def flatten(self, x):
        """
        :param x:(batch, num_frames, num_instruments, ...)
        :return: (batch, num_frames * num_instruments, ...) with num_instruments varying faster
        """
        batch_size = x.size()[0]
        x = torch.reshape(x, (batch_size, self.num_frames * self.num_ticks_per_frame, -1))
        return x

    def wrap(self, flatten_x):
        """
        Inverse of flatten operation

        :param flatten_x: (batch_size, length, ...)
        :return:
        """
        batch_size = flatten_x.size(0)
        x = torch.reshape(flatten_x, (batch_size, self.num_frames, self.num_ticks_per_frame, -1))
        return x

    def mask_encoder(self, x, p):
        return x, None

    def mask_decoder(self, x, p):
        #  Here the goal of masking is to prevent model to rely too much on past orchestra,
        # and force it to use piano information more
        if p > 0:
            x, mask = self.mask_STANDARD(x, p, epoch_id=0)
        else:
            x = x
            mask = None
        return x, mask

    def mask_nade(self, x, epoch_id=None):
        return self.mask_UNIFORM_ORDER(x, epoch_id)

    def mask_STANDARD(self, x, p, epoch_id=None):

        #  Sample an ordering (size batch_size)

        if p is None:
            #  Shit scheduling, should rather be based on the error decrement
            # Here we reach stable state after 10 epochs
            if epoch_id is not None:
                amplitude = min(epoch_id + 1, 5) / 5.
                mean_proba = min(epoch_id + 1, 5) / 10.
            else:
                amplitude = 1.0
                mean_proba = 0.5
            p = (random.random() - 0.5) * amplitude + mean_proba

        batch_size, num_frames, num_instruments = x.size()
        if epoch_id > 5:
            #  Stop learning to predict the past orchestra frames
            rand_part = (torch.rand(batch_size, 1, num_instruments).float() < p).long()
            zero_part = torch.zeros(batch_size, num_frames - 1, num_instruments).long()
            mask = torch.cat([zero_part, rand_part], dim=1).cuda()
        else:
            mask = (torch.rand_like(x.float()) < p).long()  #  1 means masked

        mask_value_matrix = self.dataset.precomputed_vectors_orchestra[MASK_SYMBOL].cuda()
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(num_frames, 1)
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        masked_chorale = x.clone() * (1 - mask) + mask_value_matrix.clone() * mask
        return masked_chorale, mask

    def mask_UNIFORM_ORDER(self, x, epoch_id=None):
        # Order is sampled first, then mask built a s permutation
        # Masking done on all frames, no scheduling...
        batch_size, num_frames, num_instruments = x.size()

        # Order is sampled uniformly
        length_orchestra = num_frames * num_instruments
        masks_non_shuffled = np.zeros((batch_size, length_orchestra))
        num_masked_events = np.random.randint(low=1, high=length_orchestra + 1, size=(batch_size))  #   0 is useless
        for batch_index in range(batch_size):
            masks_non_shuffled[batch_index, :num_masked_events[batch_index]] = 1

        mask_np = (np.random.permutation(masks_non_shuffled.T)).T
        mask_reshape = mask_np.reshape((batch_size, num_frames, num_instruments))

        if epoch_id > 5:
            # Only allow masking in the last frame
            mask_reshape[:, :-1] = 0

        mask = torch.tensor(mask_reshape).long().cuda()  #  1 means masked

        mask_value_matrix = torch.Tensor(self.num_token_per_tick).long().cuda()
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(num_frames, 1)
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        masked_chorale = x.clone() * (1 - mask) + mask_value_matrix.clone() * mask
        return masked_chorale, mask

    def mask_GEOMETRIC_SCHEDULING(self, x, p=None, epoch_id=None):
        # Order is sampled first, then mask built a s permutation
        # n ~ geometric
        # Little scheduling, after step 5, mask and thus train only on the last frame

        batch_size, num_frames, num_instruments = x.size()

        # Order is sampled uniformly
        length_orchestra = num_frames * num_instruments
        masks_non_shuffled = np.zeros((batch_size, length_orchestra))

        # Sample from linearly decreasing discrete proba (fastest method)
        S = np.arange(1, length_orchestra + 1).sum()
        orders = []
        while len(orders) < batch_size:
            n = random.randint(1, length_orchestra)
            p = n / S
            e = random.random()
            if e < p:
                orders.append(n)

        # aa = np.histogram(orders, bins=length_orchestra, range=(1, length_orchestra))
        # import matplotlib
        # import matplotlib.pyplot as plt
        # plt.bar(aa[1][:-1], aa[0])
        # x = np.linspace(1, length_orchestra, 10000)
        # plt.plot(x, aa[0][0]/x)
        # plt.savefig('test.pdf')

        for batch_index in range(batch_size):
            masks_non_shuffled[batch_index, :orders[batch_index]] = 1

        mask_np = (np.random.permutation(masks_non_shuffled.T)).T
        mask_reshape = mask_np.reshape((batch_size, num_frames, num_instruments))
        if epoch_id > 5:
            # Only allow masking in the last frame
            mask_reshape[:, :-1] = 0

        mask = torch.tensor(mask_reshape).long().cuda()  #  1 means masked

        mask_value_matrix = torch.Tensor(self.num_token_per_tick).long().cuda()
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(num_frames, 1)
        mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        #  Todo: PROBLEM equivalent to using 0 as a padding information
        masked_chorale = x.clone() * (1 - mask) + mask_value_matrix.clone() * mask
        return masked_chorale, mask

    # TODO Not implemented
    def mask_instrument_activations(self, x, p=None):
        raise Exception('Not implemented!')
        # Masking is done instrument-wise
        # i.e. if we deiced to mask an instrument, it is for all time frames
        batch_size, num_frames, num_instruments = x.size()

        # Order is sampled uniformly
        masks_non_shuffled = np.zeros((batch_size, num_instruments))
        num_masked_events = np.random.randint(low=1, high=num_instruments + 1, size=(batch_size))  #   0 is useless
        for batch_index in range(batch_size):
            masks_non_shuffled[batch_index, :num_masked_events[batch_index]] = 1

        mask_np = (np.random.permutation(masks_non_shuffled.T)).T

        mask_flat = torch.tensor(mask_np).long().cuda()  #  1 means masked
        mask = mask_flat.unsqueeze(1).repeat(1, num_frames, 1)

        pad_matrix = self.dataset.precomputed_vectors_orchestra_instruments_presence[UNKNOWN_SYMBOL].long().cuda()
        pad_matrix = pad_matrix.unsqueeze(0).repeat(num_frames, 1)
        pad_matrix = pad_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        masked_chorale = x.clone() * (1 - mask) + pad_matrix.clone() * mask
        return masked_chorale, mask

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return: preds: (num_instru, batch, num_frames, num_pitches)
        """
        batch_size, length, num_features = pred_seq.size()

        assert length % self.num_instruments == 0
        # split voices
        pred_seq = pred_seq.view(batch_size, length // self.num_instruments,
                                 self.num_instruments, num_features).permute(2, 0, 1, 3)
        preds = [
            pre_softmax(pred)
            for pred, pre_softmax in zip(pred_seq, self.linear_output_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param batch_size:
        :param sequence_length:
        :return:
        """
        num_repeats = (sequence_length + 1) // self.num_ticks_per_frame
        positions = self.instrument_labels.unsqueeze(0).repeat(batch_size, num_repeats)[:, :sequence_length]
        embedded_positions = self.instrument_embedding(positions)
        return embedded_positions

    def get_relative_attention_module(self, embedding_dim,
                                      n_head,
                                      use_masks,
                                      shift,
                                      enc_dec_attention):
        if self.flag == 'orchestra':
            if enc_dec_attention == 'cond':
                # if self.block_attention:
                #     assert use_masks is False, "Use mask should be false for encoder-decoder self-attention"
                #     return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                #                                     n_head=n_head,
                #                                     dim_in=self.num_instruments,
                #                                     dim_out=self.num_pitch_piano,
                #                                     num_frames_in=self.num_frames_orchestra,
                #                                     num_frames_out=self.num_frames_piano,
                #                                     use_masks=use_masks,
                #                                     use_instrument_attention=False,
                #                                     shift=shift)
                # else:

                #  Never use self attention matrices for encoder-decoder attentions (use zeros)
                #  Encoder-decoder attention
                if self.double_conditioning == 'concatenate':
                    seq_len_encoder = self.max_len_sequence_encodencoder + self.max_len_sequence_encoder
                else:
                    seq_len_encoder = self.max_len_sequence_encoder
                return NoPositionwiseAttention(seq_len_enc=seq_len_encoder)
            elif enc_dec_attention == 'double_cond':
                #  Case of stacked conditioning
                assert self.double_conditioning == 'stack_conditioning_layer'
                seq_len_encoder = self.max_len_sequence_encodencoder
                return NoPositionwiseAttention(seq_len_enc=seq_len_encoder)
            elif enc_dec_attention is None:
                # Decoder attention
                if self.block_attention:
                    return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                                                    n_head=n_head,
                                                    dim_in=self.num_instruments,
                                                    dim_out=self.num_instruments,
                                                    num_frames_in=self.num_frames_orchestra,
                                                    num_frames_out=self.num_frames_orchestra,
                                                    use_masks=use_masks,
                                                    use_voice_attention=True,
                                                    shift=shift)
                else:
                    return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                                       n_head=n_head,
                                                       seq_len=self.max_len_sequence_decoder,
                                                       use_masks=use_masks)
            else:
                raise ValueError()
        elif self.flag == 'piano':
            if enc_dec_attention:
                #  WARNING: it's encoder and decoder here are swaped on purpose
                seq_len_encoder = self.max_len_sequence_encodencoder
                return NoPositionwiseAttention(seq_len_enc=seq_len_encoder)
            else:
                if self.block_attention:
                    return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                                                    n_head=n_head,
                                                    dim_in=self.num_pitch_piano,
                                                    dim_out=self.num_pitch_piano,
                                                    num_frames_in=self.num_frames_piano,
                                                    num_frames_out=self.num_frames_piano,
                                                    use_masks=False,
                                                    use_voice_attention=True,
                                                    shift=False)
                else:
                    return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                                       n_head=n_head,
                                                       seq_len=self.max_len_sequence_encoder,
                                                       use_masks=use_masks)

        elif self.flag == 'instruments':
            return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                               n_head=n_head,
                                               seq_len=self.max_len_sequence_encodencoder,
                                               use_masks=use_masks)

    def extract_context_for_generation(self, frame_index, context_size, matrix):
        start_frame = frame_index - context_size
        end_frame = frame_index + context_size
        if self.flag == 'piano':
            extracted_context = matrix[:, start_frame:end_frame + 1, :]
        elif self.flag == 'orchestra':
            extracted_context = matrix[:, start_frame:end_frame + 1, :]
        elif self.flag == 'instruments':
            # Because during training instruemnt presence is computed over a window, we have to cheat a litlle bit here
            extracted_context = matrix[:, frame_index, :].unsqueeze(1).repeat(1, end_frame - start_frame + 1, 1)
        return extracted_context

    def get_range_generation(self, context_size, number_piano_frames_to_orchestrate):
        first_frame = context_size
        last_frame = number_piano_frames_to_orchestrate - 1 - context_size
        events = range(first_frame, last_frame + 1)
        return events
