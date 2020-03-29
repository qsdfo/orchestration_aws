import torch
from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from torch import nn
from Transformer.arrangement.arrangement_data_processor import ArrangementDataProcessor


class ArrangementVoiceDataProcessor(ArrangementDataProcessor):
    def __init__(self, dataset: ArrangementVoiceDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 flag,
                 block_attention,
                 nade,
                 double_conditioning
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
        # Todo: Call super only on DataProcessor. Could call it directly on super,
        #  but it would need to be further factorised by modifying the dataset classes (ArrangementDataset)
        super(ArrangementDataProcessor, self).__init__(dataset=dataset,
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
        self.num_voices_piano = dataset.number_voices_piano

        self.num_notes_per_voice_orchestra = [len(v) for k, v in self.dataset.index2midi_pitch.items()]
        self.num_notes_per_voice_piano = [len(self.dataset.index2midi_pitch_piano)] * self.num_voices_piano

        # Generic names
        self.max_len_sequence_decoder = self.num_instruments * self.num_frames_orchestra
        self.max_len_sequence_encoder = self.num_voices_piano * self.num_frames_piano
        self.max_len_sequence_encodencoder = dataset.instrument_presence_dim
        if self.flag == 'orchestra':
            self.num_token_per_tick = self.num_notes_per_voice_orchestra
            self.num_ticks_per_frame = self.num_instruments
            self.num_frames = self.num_frames_orchestra
            max_len_sequence = self.max_len_sequence_decoder
        elif self.flag == 'piano':
            #  2 options: a different embedding for each voice...
            self.num_token_per_tick = self.num_notes_per_voice_piano
            self.num_ticks_per_frame = self.num_voices_piano
            self.num_frames = self.num_frames_piano
            max_len_sequence = self.max_len_sequence_encoder
        elif self.flag == 'instruments':
            self.num_token_per_tick = [len(dataset.index2instruments_presence)] * dataset.instrument_presence_dim
            self.num_ticks_per_frame = dataset.instrument_presence_dim
            self.num_frames = 1
            max_len_sequence = self.max_len_sequence_encodencoder
        else:
            raise Exception('Unrecognized data processor flag')

        self.reducer_input_dim = reducer_input_dim
        self.local_position_embedding_dim = local_position_embedding_dim

        # Todo:
        #  In arrangement_voice_data_processor and arrangement_midi_data_processor,
        #  use the same embeddings and linear mapping for all voices, which represent the same spaces.
        #  if self.flag = 'instruments' or 'piano'... write different embeddings

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

        self.name = 'arrangement_voice_data_processor'
        return
