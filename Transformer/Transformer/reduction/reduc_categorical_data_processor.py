import music21
import numpy as np
import torch
from torch import nn

from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from DatasetManager.helpers import REST_SYMBOL, END_SYMBOL, START_SYMBOL

from Transformer.relative_attentions import NoPositionwiseAttention, RelativeSelfAttentionModule, \
    BlockSelfAttentionModule
from Transformer.data_processor import DataProcessor
from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll


class ReductionCategoricalDataProcessor(DataProcessor):
    def __init__(self, dataset: ArrangementVoiceDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 flag,
                 block_attention,
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
        super(ReductionCategoricalDataProcessor, self).__init__(dataset=dataset,
                                                                embedding_dim=embedding_dim)

        self.dataset = dataset
        self.flag = flag
        self.block_attention = block_attention
        self.flip_masks = False
        if flag == 'piano':
            self.use_masks = True
        else:
            self.use_masks = False

        # Useful parameters
        self.num_frames_orchestra = dataset.sequence_size
        self.num_instruments = dataset.number_instruments
        #
        self.num_frames_piano = int(
            (dataset.sequence_size + 1) / 2)  # Only past and present is used for the orchestration
        self.number_voices_piano = dataset.number_voices_piano

        self.num_notes_per_instrument_orchestra = [len(v) for k, v in self.dataset.index2midi_pitch.items()]
        self.num_notes_per_instrument_piano = [len(v) for k, v in self.dataset.index2midi_pitch_piano.items()]

        # Generic names
        self.max_len_sequence_decoder = self.number_voices_piano * self.num_frames_piano
        self.max_len_sequence_encoder = self.num_instruments * self.num_frames_orchestra
        if self.flag == 'orchestra':
            self.num_token_per_tick = self.num_notes_per_instrument_orchestra
            self.num_ticks_per_frame = self.num_instruments
            self.num_frames = self.num_frames_orchestra
            max_len_sequence = self.max_len_sequence_encoder
        elif self.flag == 'piano':
            self.num_token_per_tick = self.num_notes_per_instrument_piano
            self.num_ticks_per_frame = self.number_voices_piano
            self.num_frames = self.num_frames_piano
            max_len_sequence = self.max_len_sequence_decoder
        else:
            raise Exception('Unrecognized data processor flag')

        self.reducer_input_dim = reducer_input_dim
        self.local_position_embedding_dim = local_position_embedding_dim

        #  +1 is for masking
        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + 1, self.embedding_dim)
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
        return

    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        piano, orchestra, _ = tensors
        # Here extract information used by the network
        if self.flag == 'orchestra':
            return orchestra.long().cuda(non_blocking=True)
        elif self.flag == 'piano':
            # Use only past and present
            t = (self.num_frames_orchestra + 1) // 2
            tmn = t - self.num_frames_piano
            piano_past_present = piano[:, tmn:t]
            return piano_past_present.long().cuda(non_blocking=True)
        else:
            raise Exception('Unrecognized data processor flag')

    def get_len_max_seq(self):
        return self.max_len_sequence_encoder if self.flag == 'orchestra' else self.max_len_sequence_decoder

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
            #  Deal with the case of single piano frame
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
        if p != 0:
            batch_size, num_frames, num_notes = x.size()
            mask = (torch.rand_like(x.float()) < p).long()

            mask_value_matrix = cuda_variable(torch.Tensor(self.num_token_per_tick).long())
            mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(num_frames, 1)
            mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

            ret = x.clone() * (1 - mask) + mask_value_matrix.clone() * mask
        else:
            ret = x
            mask = None
        return ret, mask

    def mask_decoder(self, x, p):
        if p != 0:
            batch_size, num_frames, num_notes = x.size()
            mask = (torch.rand_like(x.float()) < p).long()

            mask_value_matrix = cuda_variable(torch.Tensor(self.num_token_per_tick).long())
            mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(num_frames, 1)
            mask_value_matrix = mask_value_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

            ret = x.clone() * (1 - mask) + mask_value_matrix.clone() * mask
        else:
            ret = x
            mask = None
        return ret, mask

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return: preds: (num_instru, batch, num_frames, num_pitches)
        """
        batch_size, length, num_features = pred_seq.size()

        assert length % self.num_ticks_per_frame == 0
        # split voices
        pred_seq = pred_seq.view(batch_size, length // self.num_ticks_per_frame,
                                 self.num_ticks_per_frame, num_features).permute(2, 0, 1, 3)
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
                                      len_max_seq_cond,
                                      len_max_seq,
                                      use_masks,
                                      shift,
                                      enc_dec_attention):
        if self.flag == 'piano':
            if enc_dec_attention:
                seq_len_encoder = len_max_seq_cond
                return NoPositionwiseAttention(seq_len_enc=seq_len_encoder)
                # return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                #                                    n_head=n_head,
                #                                    seq_len=len_max_seq,
                #                                    use_masks=use_masks)
            else:
                # Decoder attention
                if self.block_attention:
                    return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                                                    n_head=n_head,
                                                    dim_in=self.num_pitch_piano,
                                                    dim_out=self.num_pitch_piano,
                                                    num_frames_in=self.num_frames_piano,
                                                    num_frames_out=self.num_frames_piano,
                                                    use_masks=use_masks,
                                                    use_voice_attention=True,
                                                    shift=shift)
                else:
                    return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                                       n_head=n_head,
                                                       seq_len=len_max_seq,
                                                       use_masks=use_masks)
        elif self.flag == 'orchestra':
            if self.block_attention:
                return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                                                n_head=n_head,
                                                dim_in=self.num_instruments,
                                                dim_out=self.num_instruments,
                                                num_frames_in=self.num_frames_orchestra,
                                                num_frames_out=self.num_frames_orchestra,
                                                use_masks=False,
                                                use_voice_attention=True,
                                                shift=False)
            else:
                return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                                   n_head=n_head,
                                                   seq_len=len_max_seq,
                                                   use_masks=use_masks)

    def init_reduction_filepath(self, batch_size, filepath, banned_instruments=[], unknown_instruments=[],
                                subdivision=None):

        context_size = self.num_frames_piano - 1

        # Get pianorolls
        score_orchestra = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.dataset.subdivision

        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(score_orchestra,
                                                                      subdivision,
                                                                      self.dataset.simplify_instrumentation,
                                                                      self.dataset.instrument_grouping,
                                                                      self.dataset.transpose_to_sounding_pitch)

        #  New events orchestra
        onsets_cumulated = None
        for k, v in onsets_orchestra.items():
            if onsets_cumulated is None:
                onsets_cumulated = v.sum(1)
            else:
                onsets_cumulated += v.sum(1)
        rhythm_orchestra = np.where(onsets_cumulated > 0)[0]

        orchestra_tensor = []
        rhythm_orchestra_clean = []
        for frame_index in rhythm_orchestra:
            orchestra_t_encoded, _ = self.dataset.pianoroll_to_orchestral_tensor(
                pianoroll_orchestra,
                onsets_orchestra,
                frame_index)
            if orchestra_t_encoded is not None:
                orchestra_tensor.append(orchestra_t_encoded)
                rhythm_orchestra_clean.append(frame_index)

        # Prepend rests frames at the beginning and end of the piano score
        orchestra_tensor = [self.dataset.precomputed_vectors_orchestra[START_SYMBOL]] * context_size + \
                           orchestra_tensor + \
                           [self.dataset.precomputed_vectors_orchestra[END_SYMBOL]] * context_size
        orchestra_init = torch.stack(orchestra_tensor)

        # Orchestra
        num_frames = orchestra_init.shape[0]  #  Here batch size is time dimensions (each batch index is a piano event)
        piano_init = self.dataset.precomputed_vectors_piano[REST_SYMBOL]
        piano_init = piano_init.unsqueeze(0).repeat(num_frames, 1)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)

        return cuda_variable(piano_init.long()), rhythm_orchestra_clean, cuda_variable(orchestra_init.long())