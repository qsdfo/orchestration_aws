import random

import numpy as np
import torch
from DatasetManager.helpers import REST_SYMBOL, END_SYMBOL, PAD_SYMBOL, START_SYMBOL
from torch import nn

from DatasetManager.chorale_dataset import ChoraleBeatsDataset

from Transformer.relative_attentions import RelativeSelfAttentionModule, NoPositionwiseAttention
from Transformer.data_processor import DataProcessor


class BachBeatsDataProcessor(DataProcessor):
    def __init__(self, dataset: ChoraleBeatsDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 encoder_flag,
                 monophonic_flag,
                 nade_flag
                 ):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """
        super(BachBeatsDataProcessor, self).__init__(dataset=dataset,
                                                     embedding_dim=embedding_dim)
        self.dataset = dataset

        # Useful because the data_processor will slightly change in that case
        self.nade_flag = nade_flag

        #  This flag allows to feed only a melody to the encoder.
        # Set it False even for the encoder to feed the whole chorale as conditioning information
        self.encoder_flag = encoder_flag
        self.monophonic_flag = monophonic_flag

        ################################
        # notes: Here to modify the way the masks are applied to the encoder and decoder.
        #  Depend on nade or not, do we want the encoder to access anything

        # notes:
        #  attention anti-causal for encoder
        #  attention causal for decoder
        # if encoder_flag:
        #     self.flip_masks = True
        # else:
        #     self.flip_masks = False
        # self.use_masks = True

        # notes:
        #  attention complete for encoder
        #  attention causal for decoder, unless it's nade
        self.flip_masks = False
        if encoder_flag or nade_flag:
            self.use_masks = False
        else:
            self.use_masks = True
        ################################

        self.num_notes_per_voice = [len(d) for d in self.dataset.note2index_dicts]
        if monophonic_flag:
            self.num_notes_per_voice = [self.num_notes_per_voice[0]]

        self.num_voices = len(self.num_notes_per_voice)
        self.local_position_embedding_dim = local_position_embedding_dim
        max_len_sequence = self.dataset.sequences_size * self.dataset.subdivision * self.num_voices
        self.max_len_sequence_encoder = max_len_sequence
        if not nade_flag:
            self.max_len_sequence_decoder = max_len_sequence // 2
        else:
            self.max_len_sequence_decoder = max_len_sequence

        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + 1, self.embedding_dim)
                for num_embeddings in self.num_notes_per_voice
            ]
        )

        self.linear_ouput_notes = nn.ModuleList(
            [
                nn.Linear(reducer_input_dim, num_notes)
                for num_notes in self.num_notes_per_voice
            ]
        )

        # for local position
        self.num_local_positions = self.num_voices * self.dataset.subdivision
        self.instrument_labels = nn.Parameter(
            torch.Tensor([(i % self.num_local_positions) for i in range(
                self.num_local_positions)]).long(),
            requires_grad=False)

        self.instrument_embedding = nn.Embedding(self.num_local_positions,
                                                 local_position_embedding_dim)

    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        if self.nade_flag:
            ret = tensors[0]
        elif self.encoder_flag:
            #  Use only the melody ?
            # ret = tensors[0][:, 0:1, :]
            ret = tensors[0]
        else:
            batch, voices, length = tensors[0].size()
            ret = tensors[0][:, :, :length // 2]
        return ret.long().cuda(non_blocking=True), None

    @staticmethod
    def prepare_target_for_loss(target):
        return target.permute(0, 2, 1)

    @staticmethod
    def prepare_mask_for_loss(mask):
        return mask.permute(0, 2, 1)

    def embed(self, x):
        """
        :param x: (batch_size, num_voices, chorale_length)
        :return: seq: (batch_size, chorale_length * num_voices, embedding_size)
        """
        separate_voices = x.split(split_size=1, dim=1)
        separate_voices = [
            embedding(voice[:, 0, :])[:, None, :, :]
            for voice, embedding
            in zip(separate_voices, self.note_embeddings)
        ]
        x = torch.cat(separate_voices, 1)
        x = self.flatten(x=x)
        return x

    def flatten(self, x):
        """

        :param x:(batch, num_voices, chorale_length, ...)
        :return: (batch, num_voices * chorale_length, ...) with num_voices varying faster
        """
        size = x.size()
        assert len(size) >= 3
        batch_size, num_voices, chorale_length = size[:3]
        remaining_dims = list(size[3:])
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_voices * chorale_length, *remaining_dims)
        return x

    def mask_encoder(self, x, p):
        #  Encoder part, we just want to remove information, randomly
        return self.mask_UNIFORM_ORDER(x)

    def mask_decoder(self, x, p):
        # This is more like input dropout, to enforce de decoder to use the encoder information
        return self.mask(x, p)

    # TODO Not implemented!!!
    def mask_nade(self, x, epoch_id):
        return self.mask_UNIFORM_ORDER(x)

    def mask_UNIFORM_ORDER(self, x):
        # Order is sampled first, then mask built as permutation
        # Masking done on all frames, no scheduling...
        batch_size, num_voices, length = x.size()

        # Order is sampled uniformly
        flat_dim = num_voices * length
        masks_non_shuffled = np.zeros((batch_size, flat_dim))
        lower_proba = int(flat_dim // 3)
        higher_proba = flat_dim + 1
        num_masked_events = np.random.randint(low=lower_proba, high=higher_proba, size=(batch_size))  #   0 is useless
        for batch_index in range(batch_size):
            masks_non_shuffled[batch_index, :num_masked_events[batch_index]] = 1

        mask_np = (np.random.permutation(masks_non_shuffled.T)).T
        mask_reshape = mask_np.reshape((batch_size, num_voices, length))
        mask = torch.tensor(mask_reshape).long().cuda()  #  1 means masked

        # Mask symbol is the last token (self.num_notes_per_voice)
        mask_symbol_matrix = torch.Tensor(self.num_notes_per_voice).long().cuda()
        mask_symbol_matrix = mask_symbol_matrix.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, length)

        masked_chorale = x.clone() * (1 - mask) + mask_symbol_matrix.clone() * mask
        return masked_chorale, mask

    def mask(self, x, p):
        if p is None:
            p = random.random() / 2 + 0.5

        batch_size, num_voices, length = x.size()
        mask = (torch.rand_like(x.float()) < p).long()
        nc_indexes = torch.Tensor(self.num_notes_per_voice).long().cuda()
        nc_indexes = nc_indexes.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, length)

        masked_chorale = x.clone() * (1 - mask) + nc_indexes * mask
        return masked_chorale, mask

    @staticmethod
    def mean_crossentropy(pred, target, ratio=None):

        """

        :param pred: list (batch, chorale_length, num_notes) one for each voice
        since num_notes are different
        :param target:(batch, voice, chorale_length)
        :return:
        """
        # TODO use ratio argument?
        cross_entropy = nn.CrossEntropyLoss(size_average=True)
        sum = 0
        # put voice first for targets
        targets = target.permute(1, 0, 2)
        for voice_index, (voice_weight, voice_target) in enumerate(zip(pred, targets)):
            # put time first
            voice_weight = voice_weight.permute(1, 0, 2)
            voice_target = voice_target.permute(1, 0)
            for time_index, (w, t) in enumerate(zip(voice_weight, voice_target)):
                if time_index == 0 and voice_index == 0:  # exclude first dummy prediction
                    # if time_index < 12: # exclude first predictions
                    continue
                ce = cross_entropy(w, t)
                sum += ce

        return sum

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return:
        """
        batch_size, length, num_features = pred_seq.size()

        assert length % 4 == 0
        # split voices
        pred_seq = pred_seq.view(batch_size,
                                 length // 4, 4, num_features).permute(0, 2, 1, 3)
        # pred_seq (b, num_voices, chorale_length, num_features)
        preds = pred_seq.split(1, dim=1)
        preds = [
            pre_softmax(pred[:, 0, :, :])
            for pred, pre_softmax in zip(preds, self.linear_ouput_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param sequence_length:
        :return:
        """
        num_repeats = sequence_length // self.num_local_positions + 1
        positions = self.instrument_labels.unsqueeze(0).repeat(batch_size, num_repeats)[:,
                    :sequence_length]
        embedded_positions = self.instrument_embedding(positions)
        return embedded_positions

    def get_relative_attention_module(self, embedding_dim,
                                      n_head,
                                      use_masks,
                                      shift,
                                      enc_dec_attention,
                                      ):

        if enc_dec_attention:
            seq_len_encoder = self.max_len_sequence_encoder
            return NoPositionwiseAttention(seq_len_enc=seq_len_encoder)
            # return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
            #                                    n_head=n_head,
            #                                    seq_len=len_max_seq,
            #                                    use_masks=use_masks)
        else:
            if self.encoder_flag:
                seq_len = self.max_len_sequence_encoder
            else:
                seq_len = self.max_len_sequence_decoder
            return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                               n_head=n_head,
                                               seq_len=seq_len,
                                               use_masks=use_masks)

    def init_generation(self, num_measures=12, ascii_melody=None, append_beginning=16, append_end=32):
        if ascii_melody is None:
            start_notes = torch.Tensor([[
                [self.dataset.note2index_dicts[0]['START']],
                [self.dataset.note2index_dicts[1]['START']],
                [self.dataset.note2index_dicts[2]['START']],
                [self.dataset.note2index_dicts[3]['START']]
            ]]).long()
            chorale = start_notes.repeat(
                1, 1, num_measures * 4 * self.dataset.subdivision)

            no_constraints_notes = torch.Tensor([[
                [len(self.dataset.note2index_dicts[0])],
                [len(self.dataset.note2index_dicts[1])],
                [len(self.dataset.note2index_dicts[2])],
                [len(self.dataset.note2index_dicts[3])]
            ]]).long()
            constraint_chorale = no_constraints_notes.repeat(
                1, 1, num_measures * 4 * self.dataset.subdivision
            )
        else:
            constraint_chorale = self.constraint_chorale_from_ascii_melody(ascii_melody)

        start_notes = torch.Tensor([[
            [self.dataset.note2index_dicts[0][START_SYMBOL]],
            [self.dataset.note2index_dicts[1][START_SYMBOL]],
            [self.dataset.note2index_dicts[2][START_SYMBOL]],
            [self.dataset.note2index_dicts[3][START_SYMBOL]]
        ]]).long()

        pad_notes = torch.Tensor([[
            [self.dataset.note2index_dicts[0][PAD_SYMBOL]],
            [self.dataset.note2index_dicts[1][PAD_SYMBOL]],
            [self.dataset.note2index_dicts[2][PAD_SYMBOL]],
            [self.dataset.note2index_dicts[3][PAD_SYMBOL]]
        ]]).long()

        end_notes = torch.Tensor([[
            [self.dataset.note2index_dicts[0][END_SYMBOL]],
            [self.dataset.note2index_dicts[1][END_SYMBOL]],
            [self.dataset.note2index_dicts[2][END_SYMBOL]],
            [self.dataset.note2index_dicts[3][END_SYMBOL]]
        ]]).long()

        melody_length = len(ascii_melody)
        chorale_begin = torch.cat([pad_notes.repeat(1, 1, append_beginning-1), start_notes], dim=2)
        chorale_end = torch.cat([end_notes, pad_notes.repeat(1, 1, append_beginning - 1)], dim=2)
        chorale_pad = start_notes.repeat(1, 1, melody_length)

        #   Cat or replace ?
        constraint_chorale = torch.cat([chorale_begin,
                                        constraint_chorale,
                                        chorale_end], dim=2)
        chorale = torch.cat([chorale_begin,
                             chorale_pad,
                             chorale_end], dim=2)

        return chorale.cuda(), constraint_chorale.cuda()

    def constraint_chorale_from_ascii_melody(self, ascii_melody):
        melody_indexes = torch.Tensor([self.dataset.note2index_dicts[0][n]
                                       if not ((n == 'NC') or (n not in self.dataset.note2index_dicts[0].keys())) else
                                       self.dataset.note2index_dicts[0][REST_SYMBOL]
                                       for n in ascii_melody]).long().unsqueeze(0).unsqueeze(0)
        no_constraints_other_voices = torch.Tensor([[
            [len(self.dataset.note2index_dicts[1])],
            [len(self.dataset.note2index_dicts[2])],
            [len(self.dataset.note2index_dicts[3])]
        ]]).long()
        melody_length = len(ascii_melody)
        other_voices = no_constraints_other_voices.repeat(1, 1, melody_length)
        constraint_chorale = torch.cat(
            [melody_indexes, other_voices], 1
        )
        return constraint_chorale
