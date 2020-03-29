import random

from DatasetManager.lsdb.lsdb_dataset import LsdbDataset

from Transformer.data_processor import DataProcessor
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LsdbDataProcessor(DataProcessor):
    def __init__(self, dataset: LsdbDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 ):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """
        # todo note: reducer_input_dim = d_model
        # local_position_dim = d_ticks
        super(LsdbDataProcessor, self).__init__(dataset=dataset,
                                                embedding_dim=embedding_dim)
        unary_constraint_size = 1
        self.dataset = dataset

        self.num_tokens_per_voice = [len(d)
                                     for d in self.dataset.symbol2index_dicts
                                     ]

        # use also note_embeddings to embed unary constraints
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + unary_constraint_size,
                             self.embedding_dim)
                for num_embeddings in self.num_tokens_per_voice
            ]
        )
        self.linear_output_notes = nn.ModuleList(
            [
                nn.Linear(reducer_input_dim, num_tokens)
                for num_tokens in self.num_tokens_per_voice
            ]
        )

        # for local position
        self.num_local_positions = 6 + 2
        self.local_position_embedding_dim = local_position_embedding_dim
        # self.instrument_labels = cuda_variable(
        #     torch.Tensor(
        #         [i % self.num_local_positions
        #          for i in range(self.num_local_positions)]).long())
        self.instrument_labels = nn.Parameter(
            torch.Tensor(
                [i % self.num_local_positions
                 for i in range(self.num_local_positions)]).long(),
            requires_grad=False)
        self.instrument_embedding = nn.Embedding(self.num_local_positions,
                                                 local_position_embedding_dim)
        # self.register_buffer("")

        # TODO add this in Datasets
        # self.num_tokens_per_beat = self.dataset.subdivision + 2

    def embed(self, x):
        """
        :param x: leadsheet = (lead,
                          chord_roots,
                          chord_types)
        :return:
        """
        # embed
        leadsheet_embedded = [
            embedding(part)
            for part, embedding
            in zip(x, self.embeddings)
        ]

        # flatten
        flatten_leadsheet = self.flatten(leadsheet_embedded)
        return flatten_leadsheet

    def get_len_max_seq(self):
        return self.dataset.sequences_size * (self.dataset.subdivision + 2)

    def flatten(self, x):
        """

        :param x:leadsheet = (lead,
                          chord_roots,
                          chord_types)
                          embedded or not
        :return: (batch, t, ...)
        """
        subdivision = self.dataset.subdivision
        flatten_leadsheet = []
        lead, chord_roots, chord_types = x
        num_beats = chord_roots.size(1)

        for beat_index in range(num_beats):
            flatten_leadsheet.append(
                chord_roots[:, beat_index: beat_index + 1, ...]
            )
            flatten_leadsheet.append(
                chord_types[:, beat_index: beat_index + 1, ...]
            )
            flatten_leadsheet.append(
                lead[:,
                beat_index * subdivision: (beat_index + 1) * subdivision,
                ...]
            )
        flatten_leadsheet = torch.cat(flatten_leadsheet, 1)
        return flatten_leadsheet

    def wrap(self, flatten_x):
        """
        Inverse of flatten operation

        :param flatten_x: (batch_size, length, ...)
        :return:
        """
        x = flatten_x.unfold(1,
                             size=self.dataset.subdivision + 2,
                             step=self.dataset.subdivision + 2)
        chord_roots = x[..., 0]
        chord_types = x[..., 1]
        lead = x[..., 2:]

        if len(lead.size()) == 4:
            # todo check this branch
            lead = lead.permute(0, 1, 3, 2)
            lead = lead.contiguous().view(lead.size(0), lead.size(1) * lead.size(2), -1)
        elif len(lead.size()) == 3:
            lead = lead.contiguous().view(lead.size(0), lead.size(1) * lead.size(2))
        else:
            raise NotImplementedError

        return lead, chord_roots, chord_types

    def mask(self, x, p=None):
        """

        :param x:
        :param p: p=1 -> all NC symbols
        p=0 -> x
        :return:
        """
        if p is None:
            ps = [random.random() * 0.9 for _ in range(len(x))]
            # p = random.random()
        if type(p) == float:
            ps = [p for _ in range(len(x))]
        else:
            ps = p

        masked_leadsheet = []
        constraint_location = []
        for part, symbol2index, p in zip(x, self.dataset.symbol2index_dicts, ps):
            constraint_location_part = (torch.rand_like(part.float()) < p).long()
            no_constraint_index = len(symbol2index)
            no_constraint_tensor = torch.ones_like(part) * no_constraint_index

            masked_part = (part.clone() * (1 - constraint_location_part)
                           + no_constraint_tensor * constraint_location_part
                           )
            masked_leadsheet.append(masked_part)
            constraint_location.append(constraint_location_part)
        return masked_leadsheet, constraint_location

    def mean_crossentropy(self, pred, target, ratio=1.):
        # order is (lead, chord_root, chord_type)
        # TODO only on useful area?!
        batch_size = pred[0].size(0)
        sum = 0
        for k, (weight, target_voice) in enumerate(zip(pred, target)):
            # Exclude first beat (dummy prediction)
            # chords
            if k == 1 or k == 2:
                weight = weight[:, 1: int(weight.size(1) * ratio), ...]
                target_voice = target_voice[:, 1:int(target_voice.size(1) * ratio),
                               ...]
            # lead
            else:
                weight = weight[:, self.dataset.subdivision:int(weight.size(1) * ratio), ...]
                target_voice = target_voice[:, self.dataset.subdivision:int(target_voice.size(1) * ratio), ...]

            # FIXME to remove!
            # if k == 1 or k == 2:
            #     weight = weight[:, 1:18, ...]
            #     target_voice = target_voice[:, 1:18, ...]
            # else:
            #     weight = weight[:, self.dataset.subdivision: 18 * self.dataset.subdivision, ...]
            #     target_voice = target_voice[:, self.dataset.subdivision:
            #                                    18 * self.dataset.subdivision, ...]

            # reshape if necessary

            if len(weight.size()) > 2:
                last_dim = weight.size(-1)
                weight = weight.contiguous().view(-1, last_dim)
                target_voice = target_voice.contiguous().view(-1)
            ce = F.cross_entropy(weight, target_voice, size_average=False)
            sum += ce
        return sum / batch_size

    def masked_mean_crossentropy(self, pred, target, masks):
        # order is (lead, chord_root, chord_type)
        batch_size = pred[0].size(0)
        sum = 0

        for k, (weight, target_voice, mask) in enumerate(zip(pred, target, masks)):
            # reshape if necessary

            if len(weight.size()) > 2:
                last_dim = weight.size(-1)
                weight = weight.contiguous().view(-1, last_dim)
                target_voice = target_voice.contiguous().view(-1)
                mask = mask.view(-1)
            ce = F.cross_entropy(weight,
                                 target_voice,
                                 reduction='none').masked_fill((1 - mask).byte(), 0)
            sum += ce.sum()
        return sum / batch_size

    def pred_seq_to_preds(self, pred_seq):
        """

        :param pred_seq:
        :return:
        """
        batch_size, flatten_leadsheet_length, _ = pred_seq.size()
        # todo 2 hardcoded: 1 for chord_roots, 1 for chord_types
        subdivision = self.dataset.subdivision
        num_ticks_per_beat = subdivision + 2

        assert flatten_leadsheet_length % num_ticks_per_beat == 0
        num_beats = flatten_leadsheet_length // num_ticks_per_beat

        lead_weights = []
        chord_roots_weights = []
        chord_types_weights = []

        for beat_index in range(num_beats):
            current_beat_start = beat_index * num_ticks_per_beat
            chord_roots_weights.append(
                pred_seq[:, current_beat_start: current_beat_start + 1]
            )
            chord_types_weights.append(
                pred_seq[:, current_beat_start + 1: current_beat_start + 2]
            )
            lead_weights.append(
                pred_seq[:, current_beat_start + 2: current_beat_start + 2 + subdivision]
            )
        lead_weights = torch.cat(lead_weights, 1)
        chord_roots_weights = torch.cat(chord_roots_weights, 1)
        chord_types_weights = torch.cat(chord_types_weights, 1)

        leadsheet_weights = lead_weights, chord_roots_weights, chord_types_weights

        leadsheet_weights = [
            proj(weights)
            for proj, weights in zip(self.linear_output_notes, leadsheet_weights)
        ]
        return leadsheet_weights

    def preds_to_flattened_probabilities(self, preds):
        """

        :param preds: list of (batch_size, !=length, !=alphabet_sizes)
        of presoftmax weights
        :return: (batch_size, flattened_length, max_alphabet_size) tensor
        of presoftmax weights
        """
        max_length = max([pred.size(2) for pred in preds])
        padded_preds = [F.pad(pred, (0, max_length - pred.size(2)), mode='constant', value=-np.inf)
                        for pred in preds]
        return self.flatten(padded_preds)

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
                                      max_len_seq,
                                      use_masks):
        return LsdbRelativeAttentionModule(embedding_dim=embedding_dim,
                                           n_head=n_head,
                                           seq_len=max_len_seq,
                                           use_masks=use_masks)

    def init_generation(self, num_beats=16, num_beats_context=4, num_beats_constraints=4):
        """
        :param num_beats:
        :param num_beats_context:
        :return:  (x, x_constraints) each of size (1, (num_beats + num_beats_constraints +
        num_beats_context) * 8)
        """
        pad_notes = ([self.dataset.symbol2index_dicts[1]['XX'],
                      self.dataset.symbol2index_dicts[2]['XX']] +
                     [self.dataset.symbol2index_dicts[0]['XX']] * self.dataset.subdivision
                     )
        pad_notes = torch.Tensor(pad_notes).long()

        start_notes = ([self.dataset.symbol2index_dicts[1]['START'],
                        self.dataset.symbol2index_dicts[2]['START']] +
                       [self.dataset.symbol2index_dicts[0]['XX']] *
                       (self.dataset.subdivision - 1) +
                       [self.dataset.symbol2index_dicts[0]['START']]
                       )
        start_notes = torch.Tensor(start_notes).long()

        # start_notes = ([self.dataset.symbol2index_dicts[1]['C'],
        #                 self.dataset.symbol2index_dicts[2]['m']] +
        #                [self.dataset.symbol2index_dicts[0]['__']] * self.dataset.subdivision
        #                )
        # start_notes = torch.Tensor(start_notes).long()

        # no_constraints_notes = ([len(self.dataset.symbol2index_dicts[1]),
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] * self.dataset.subdivision
        #                         )

        # only Gs
        # no_constraints_notes = ([self.dataset.symbol2index_dicts[1]['G'],
        #                          self.dataset.symbol2index_dicts[2]['7']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] * self.dataset.subdivision
        #                         )

        no_constraints_notes = ([self.dataset.symbol2index_dicts[1]['D'],
                                 self.dataset.symbol2index_dicts[2]['m7']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['D'],
                                 self.dataset.symbol2index_dicts[2]['m7']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['G'],
                                 self.dataset.symbol2index_dicts[2]['7']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['G'],
                                 self.dataset.symbol2index_dicts[2]['7']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['C'],
                                 self.dataset.symbol2index_dicts[2]['']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['C'],
                                 self.dataset.symbol2index_dicts[2]['']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['C'],
                                 self.dataset.symbol2index_dicts[2]['']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision +
                                [self.dataset.symbol2index_dicts[1]['C'],
                                 self.dataset.symbol2index_dicts[2]['']] +
                                [len(self.dataset.symbol2index_dicts[0])] *
                                self.dataset.subdivision
                                )

        # no_constraints_notes = ([self.dataset.symbol2index_dicts[1]['D'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['D'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['G'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['G'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['C'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['C'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['C'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [self.dataset.symbol2index_dicts[1]['C'],
        #                          len(self.dataset.symbol2index_dicts[2])] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision
        #                         )

        # only 7s
        # no_constraints_notes = ([len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['7']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['7']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['7']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['7']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['m']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['m']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['m']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision +
        #                         [len(self.dataset.symbol2index_dicts[1]),
        #                          self.dataset.symbol2index_dicts[2]['m']] +
        #                         [len(self.dataset.symbol2index_dicts[0])] *
        #                         self.dataset.subdivision
        #                         )

        no_constraints_notes = torch.Tensor(no_constraints_notes).long()

        end_notes = ([self.dataset.symbol2index_dicts[1]['END'],
                      self.dataset.symbol2index_dicts[2]['END']] +
                     [self.dataset.symbol2index_dicts[0]['END']] +
                     [self.dataset.symbol2index_dicts[0]['XX']] *
                     (self.dataset.subdivision - 1)
                     )
        end_notes = torch.Tensor(end_notes).long()

        x = torch.cat([
            pad_notes.repeat(num_beats_context - 1),
            start_notes,
            no_constraints_notes.repeat(num_beats // 8),
            # no_constraints_notes.repeat(num_beats),
            end_notes.repeat(num_beats_constraints),
            pad_notes.repeat(num_beats_constraints - 1)
        ])
        x = x.unsqueeze(0).cuda()
        return x, x

    def tick_index_to_dict_index(self, tick_index):
        tick_mod = tick_index % (self.dataset.subdivision + 2)
        if tick_mod == 0:
            return self.dataset.CHORD_ROOT
        elif tick_mod == 1:
            return self.dataset.CHORD_NAME
        else:
            return self.dataset.NOTES


class LsdbRelativeAttentionModule(nn.Module):
    def __init__(self, embedding_dim, n_head, seq_len, use_masks):
        # todo embedding dim = d_k
        super(LsdbRelativeAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.use_masks = use_masks
        if self.use_masks:
            self.e = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
        else:
            self.e1 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
            self.e2 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        assert sz_b_times_n_head % self.n_head == 0
        sz_b = sz_b_times_n_head // self.n_head

        if self.use_masks:
            # In this case,
            # rel_attn will be masked in RelativeScaledDotProductAttention
            # the size of e can be larger that the size of q
            # todo add shift argument?!
            e = self.e[:self.n_head * len_q, :]
            e = e.unsqueeze(0).repeat(sz_b, 1, 1)
            e = e.view(sz_b * self.n_head, len_q, d_q)

            rel_attn = torch.einsum('bld,bmd->blm',
                                    (q, e))
            batch_size, l, _ = rel_attn.size()
            # skewing trick
            if flipped_masks:
                # pad
                rel_attn = torch.cat(
                    [rel_attn,
                     (torch.ones(batch_size, l, 1, ) * - 100).cuda(),
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, :-1, :]
            else:
                # todo refine
                # pad
                rel_attn = torch.cat(
                    [(torch.ones(batch_size, l, 1, ) * - 100).cuda(),
                     rel_attn
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, 1:, :]

        # if no mask is used
        # the trick must be done twice
        else:
            e1 = self.e1.unsqueeze(0).repeat(sz_b, 1, 1)
            e1 = e1.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_1 = torch.einsum('bld,bmd->blm',
                                      (q, e1))
            e2 = self.e2.unsqueeze(0).repeat(sz_b, 1, 1)
            e2 = e2.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_2 = torch.einsum('bld,bmd->blm',
                                      (q, e2))

            batch_size, l, _ = rel_attn_1.size()
            # ====skewing trick
            # ----Down
            # pad
            rel_attn_1 = torch.cat(
                [rel_attn_1,
                 (torch.ones(batch_size, l, 1, ) * - 100).cuda(),
                 ], dim=2
            )
            rel_attn_1 = rel_attn_1.view(batch_size,
                                         l + 1,
                                         l,
                                         )

            rel_attn_1 = rel_attn_1[:, :-1, :]

            # ----Up

            # pad
            rel_attn_2 = torch.cat(
                [(torch.ones(batch_size,
                                          l,
                                          1,
                                          ) * - 100).cuda(),
                 rel_attn_2
                 ], dim=2
            )
            rel_attn_2 = rel_attn_2.view(batch_size,
                                         l + 1,
                                         l,
                                         )

            rel_attn_2 = rel_attn_2[:, 1:, :]

            masks_down = torch.triu(torch.ones_like(rel_attn_1[0]).byte(),
                                    diagonal=0).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).flip(
                1).flip(2)
            masks_up = torch.triu(torch.ones_like(rel_attn_2[0]).byte(),
                                  diagonal=1).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1)

            rel_attn_1 = rel_attn_1.masked_fill(masks_down, 0)
            rel_attn_2 = rel_attn_2.masked_fill(masks_up, 0)
            rel_attn = rel_attn_1 + rel_attn_2
        return rel_attn
