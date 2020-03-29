import random
import re

import music21
import torch
from DatasetManager.arrangement.arrangement_helper import note_to_midiPitch
from DatasetManager.helpers import START_SYMBOL, END_SYMBOL
from torch import nn

from Transformer.data_processor import DataProcessor
from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll


class ArrangementFrameDataProcessor(DataProcessor):
    def __init__(self, dataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 flag_orchestra
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
        super(ArrangementFrameDataProcessor, self).__init__(dataset=dataset,
                                                            embedding_dim=embedding_dim)
        unary_constraint_size = 1
        self.dataset = dataset
        self.flag_orchestra = flag_orchestra
        self.num_notes_per_instrument_orchestra = [len(v)
                                                   for k, v in self.dataset.index2midi_pitch.items()]
        self.num_notes_per_instrument_piano = [1 for _ in range(
            self.dataset.piano_tessitura['highest_pitch'] - self.dataset.piano_tessitura['lowest_pitch'] + 1)]
        if self.flag_orchestra:
            self.num_notes_per_instrument = self.num_notes_per_instrument_orchestra
        else:
            self.num_notes_per_instrument = self.num_notes_per_instrument_piano
        self.num_parts = len(self.num_notes_per_instrument)
        self.local_position_embedding_dim = local_position_embedding_dim

        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + unary_constraint_size, self.embedding_dim)
                for num_embeddings in self.num_notes_per_instrument
            ]
        )

        self.linear_output_notes = nn.ModuleList(
            [
                nn.Linear(reducer_input_dim, num_notes)
                for num_notes in self.num_notes_per_instrument
            ]
        )

        # for local position
        self.num_local_positions = self.num_parts
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
        if self.flag_orchestra:
            return tensors[1].long().cuda(non_blocking=True)
        else:
            # Binarize
            x = tensors[0] > 0
            return x.long().cuda(non_blocking=True)

    def get_len_max_seq(self):
        return self.num_parts

    def embed(self, x):
        """
        :param x: (batch_size, num_parts)
        :return: seq: (batch_size, num_parts, embedding_size)
        """
        x_permute = x.permute(1, 0)
        x_embeds = [
            embedding(voice)
            for voice, embedding
            in zip(x_permute, self.note_embeddings)
        ]
        x = torch.stack(x_embeds, 0)
        x = x.permute(1, 0, 2)
        return x

    def flatten(self, x):
        """
        :param x:(batch, num_voices, chorale_length, ...)
        :return: (batch, num_voices * chorale_length, ...) with num_voices varying faster
        """
        return x

    def mask(self, x, p=None):
        if p is None:
            # TODO
            # Why this super high value ???
            # p = random.random() / 2 + 0.5
            p = random.random() * 0.2 + 0.1

        batch_size, num_parts = x.size()
        mask = (torch.rand_like(x.float()) < p).long()
        nc_indexes = torch.Tensor(self.num_notes_per_instrument).long().cuda()
        nc_indexes = nc_indexes.unsqueeze(0).repeat(batch_size, 1)

        masked_chorale = x.clone() * (1 - mask) + nc_indexes * mask
        return masked_chorale, nc_indexes

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
        targets = target.permute(1, 0)
        last_voice_index = len(pred) - 1
        for voice_index, (w, t) in enumerate(zip(pred, targets)):
            if (voice_index == 0) or (voice_index == last_voice_index):
                # exclude first and last dummy prediction (Start and End symbols)
                continue
            ce = cross_entropy(w, t)
            sum += ce
        return sum

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return:
        """
        pred_seq_permute = pred_seq.permute(1, 0, 2)
        preds = [
            pre_softmax(pred)
            for pred, pre_softmax in zip(pred_seq_permute, self.linear_output_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param sequence_length:
        :return:
        """
        positions = self.instrument_labels.unsqueeze(0).repeat(batch_size, 1)
        embedded_positions = self.instrument_embedding(positions)
        return embedded_positions

    def get_relative_attention_module(self, embedding_dim,
                                      n_head,
                                      max_len_seq,
                                      use_masks,
                                      enc_dec_attention):
        if self.flag_orchestra:
            if enc_dec_attention:
                # return ArrangementAbsoluteAttentionModule
                seq_len_encoder = len(self.num_notes_per_instrument_piano)
                seq_len_decoder = len(self.num_notes_per_instrument_orchestra)
                assert (max_len_seq == seq_len_decoder)
                return PianoOrchestraRelativeAttentionModule(seq_len_enc=seq_len_encoder,
                                                             seq_len_dec=max_len_seq)
            else:
                return OrchestraRelativeAttentionModule(embedding_dim=embedding_dim,
                                                        n_head=n_head,
                                                        seq_len=max_len_seq,
                                                        use_masks=use_masks)
        else:
            return PianoRelativeAttentionModule(embedding_dim=embedding_dim,
                                                n_head=n_head,
                                                seq_len=max_len_seq,
                                                use_masks=use_masks)

    def init_generation_filepath(self, filepath, banned_instruments=[]):
        # Get pianorolls
        score_piano = music21.converter.parse(filepath)
        pianoroll_piano, _ = score_to_pianoroll(score_piano,
                                                self.dataset.subdivision,
                                                self.dataset.simplify_instrumentation,
                                                self.dataset.transpose_to_sounding_pitch)

        #  New events piano
        def new_events(pianoroll_piano):
            num_frames = pianoroll_piano.shape[0]
            events_indices = []
            frames_to_keep = []
            current_frame = None
            for frame_index in range(num_frames):
                if pianoroll_piano[frame_index].sum() == 0:
                    continue
                if (current_frame is None) or not (current_frame == pianoroll_piano[frame_index]).all():
                    events_indices.append(frame_index)
                    current_frame = pianoroll_piano[frame_index]
                    frames_to_keep.append(torch.from_numpy(current_frame))
            return events_indices, frames_to_keep

        rhythm_piano, frames_to_keep = new_events(pianoroll_piano['Piano'])
        piano_events = torch.stack(frames_to_keep)
        piano_events_reduced = piano_events[:,
                               self.dataset.piano_tessitura['lowest_pitch']: self.dataset.piano_tessitura[
                                                                                 'highest_pitch'] + 1]
        piano_events_reduced_binary = (piano_events_reduced > 0)

        # Orchestra
        batch_size = piano_events_reduced_binary.shape[
            0]  #  Here batch size is time dimensions (each batch index is a piano event)
        orchestra_silences, orchestra_init = self.init_orchestra(batch_size, banned_instruments)

        return piano_events_reduced_binary.long().cuda(), rhythm_piano, \
               orchestra_init.long().cuda(), orchestra_silences

    def init_generation(self, number_samples_per_chord=8, banned_instruments=[]):
        #  C major and inversions
        C_major = music21.chord.Chord(["C4", "E4", "G4"])
        C_major_1 = music21.chord.Chord(["E4", "G4", "C5"])
        C_major_2 = music21.chord.Chord(["G3", "C4", "E4"])
        C7 = music21.chord.Chord(["C4", "E4", "G4", "B-4"])
        # Tranpose the previous chords
        transpositions = [
            music21.interval.Interval(2),
            music21.interval.Interval(3),
            music21.interval.Interval(6),
            music21.interval.Interval(7),
        ]
        base_chords = [C_major, C_major_1, C_major_2, C7]
        transposed_chords = []
        for transposition in transpositions:
            this_transposed_chords = [chord.transpose(transposition) for chord in base_chords]
            transposed_chords.extend(this_transposed_chords)
        all_chords = base_chords + transposed_chords

        piano_dim = self.dataset.piano_tessitura['highest_pitch'] - self.dataset.piano_tessitura['lowest_pitch'] + 1
        piano_score = []
        for chord in all_chords:
            this_chord_vector = torch.zeros(piano_dim)
            for note in chord._notes:
                index = note_to_midiPitch(note) - self.dataset.piano_tessitura['lowest_pitch']
                this_chord_vector[index] = 1
            piano_score += ([this_chord_vector, ] * number_samples_per_chord)
        piano_score = torch.stack(piano_score)

        rhythm_piano = list(range(len(piano_score)))

        batch_size = piano_score.shape[0]

        orchestra_silences, orchestra_init = self.__init_orchestra(batch_size, banned_instruments)

        return piano_score.long().cuda(), rhythm_piano, \
               orchestra_init.long().cuda(), orchestra_silences

    def init_orchestra(self, batch_size, banned_instruments):
        # Set orchestra constraints in the form of banned instruments
        orchestra_silences = []
        orchestra_init = torch.zeros(batch_size, len(self.num_notes_per_instrument_orchestra))
        for instrument_name, instrument_indices in self.dataset.instrument2index.items():
            for instrument_index in instrument_indices:
                if instrument_name in banned_instruments:
                    # -1 is a silence
                    orchestra_silences.append(1)
                    orchestra_init[:, instrument_index] = self.dataset.midi_pitch2index[instrument_index][-1]
                else:
                    orchestra_silences.append(0)
                    if instrument_name in [START_SYMBOL, END_SYMBOL]:
                        orchestra_init[instrument_index] = 0
                    else:
                        #  Initialise with last
                        orchestra_init[:, instrument_index] = self.num_notes_per_instrument_orchestra[instrument_index]
        return orchestra_silences, orchestra_init


# OLD ONE
# DOES NOT USE THE TRICK FOR MAGENTA
# TODO adapt signature
# AND IT USED ONLY ONE SHARED R AMONGST THE HEADS
# class BachRelativeAttentionModule(nn.Module):
#     def __init__(self, embedding_dim, max_len_seq):
#         # todo embedding dim = d_k
#         super(BachRelativeAttentionModule, self).__init__()
#         self.embedding_dim = embedding_dim
#         # share R between heads
#         self.w_rs_tick = nn.Embedding(max_len_seq // 2 + 1, self.embedding_dim,
#                                       padding_idx=0)
#         self.w_rs_mod = nn.Embedding(17 + 1, self.embedding_dim,
#                                      padding_idx=0)
#
#         # TODO don't use triangular r?!
#         # self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + 1 if i - j >= 0 else 0 for i in
#         #                                    range(
#         #                                        get_len_max_seq)]
#         #                                   for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + max_len_seq // 4 + 1 for i in
#                                            range(max_len_seq)]
#                                           for j in range(max_len_seq)]).long().to('cuda')
#         # self.r_index_mod = torch.Tensor([[((i % 4) * 4 + (j % 4)) if i - j >= 0 else -1 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         # not the best idea:
#         undirected_order_dict = {(0, 0): 0,
#                                  (0, 1): 1,
#                                  (1, 0): 1,
#                                  (0, 2): 2,
#                                  (2, 0): 2,
#                                  (0, 3): 3,
#                                  (3, 0): 3,
#                                  (1, 1): 4,
#                                  (1, 2): 5,
#                                  (2, 1): 5,
#                                  (1, 3): 6,
#                                  (3, 1): 6,
#                                  (2, 2): 7,
#                                  (2, 3): 8,
#                                  (3, 2): 8,
#                                  (3, 3): 9,
#                                  }
#         # self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1 if i - j >= 0
#         #                                   else 0 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1
#                                           for i in
#                                           range(
#                                               max_len_seq)]
#                                          for j in range(max_len_seq)]).long().to('cuda')
#
#     def forward(self, sequence_length, n_head, batch_size):
#         # r is shared amongst the heads
#         r_mod = self.w_rs_mod(self.r_index_mod[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#         r_tick = self.w_rs_tick(self.r_index_tick[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#
#         r_mod = r_mod.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_mod = r_mod.view(-1, sequence_length, sequence_length,
#                            self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r_tick = r_tick.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_tick = r_tick.view(-1, sequence_length, sequence_length,
#                              self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r = r_mod + r_tick
#         return r

# Class copied from LsdbRelativeAttentionModule
class PianoRelativeAttentionModule(nn.Module):
    def __init__(self, embedding_dim, n_head, seq_len, use_masks):
        # todo embedding dim = d_k
        super(PianoRelativeAttentionModule, self).__init__()
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
                     cuda_variable(torch.ones(batch_size,
                                              l,
                                              1,
                                              ) * - 100),
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
                    [cuda_variable(torch.ones(batch_size,
                                              l,
                                              1,
                                              ) * - 100),
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
                 cuda_variable(torch.ones(batch_size,
                                          l,
                                          1,
                                          ) * - 100),
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
                [cuda_variable(torch.ones(batch_size,
                                          l,
                                          1,
                                          ) * - 100),
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

    def extract_context_for_generation(self, frame_index, context_size, matrix):
        start_frame = frame_index - context_size
        end_frame = frame_index + context_size
        if self.flag == 'piano':
            extracted_context = matrix[:, start_frame:end_frame + 1, :]
        elif self.flag == 'orchestra':
            extracted_context = matrix[:, start_frame:end_frame + 1, :]
        return extracted_context
