from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.arrangement.arrangement_midiPiano_dataset import ArrangementMidipianoDataset
from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from Transformer.ar.ar_data_processor import ARDataProcessor
from Transformer.ar.ar_dataset import ARDataset
from Transformer.arrangement.arrangement_data_processor import ArrangementDataProcessor
from Transformer.arrangement.arrangement_midiPiano_data_processor import ArrangementMidiPianoDataProcessor
from Transformer.arrangement.arrangement_voice_data_processor import ArrangementVoiceDataProcessor
from Transformer.bach.bach_data_processor import BachBeatsDataProcessor
from Transformer.bach.bach_data_processor_cpc import BachBeatsCPCDataProcessor
from Transformer.helpers import init_cpc_model
from Transformer.lsdb.lsdb_data_processor import LsdbDataProcessor
from Transformer.reduction.reduc_categorical_data_processor import ReductionCategoricalDataProcessor
from Transformer.reduction.reduc_data_processor import ReductionDataProcessor
from Transformer.reduction.reduc_midiPiano_data_processor import ReductionMidiPianoDataProcessor


def get_dataset(dataset_manager, dataset_type, subdivision, sequence_size, velocity_quantization, max_transposition, num_heads,
                per_head_dim, local_position_embedding_dim, block_attention, group_instrument_per_section, nade,
                cpc_config_name, double_conditioning, instrument_presence_in_encoder):
    if dataset_type == 'bach':
        if nade:
            raise Exception('j ai l impression que nade c est nimps dans le data processor; check before using')
        metadatas = [
            FermataMetadata(),
            TickMetadata(subdivision=subdivision),
            KeyMetadata()
        ]

        voices_ids = [0, 1, 2, 3]

        if cpc_config_name is not None:
            # notes to compute the first cpc code, we need to waste block_size tokens
            cpc_model = init_cpc_model(cpc_config_name)
            block_size = cpc_model.dataloader_generator.num_tokens_per_block // (subdivision * len(voices_ids))
            sequence_size += block_size

        chorale_dataset_kwargs = {
            'voice_ids': voices_ids,
            'metadatas': metadatas,
            'sequences_size': sequence_size,
            'subdivision': subdivision,
        }

        dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
            name='bach_chorales_beats',
            **chorale_dataset_kwargs
        )

        if cpc_config_name is None:
            processor_encoder = BachBeatsDataProcessor(dataset=dataset,
                                                       embedding_dim=512 - 8,
                                                       reducer_input_dim=512,
                                                       local_position_embedding_dim=8,
                                                       encoder_flag=True,
                                                       monophonic_flag=False,
                                                       nade_flag=nade)

            processor_decoder = BachBeatsDataProcessor(dataset=dataset,
                                                       embedding_dim=512 - 8,
                                                       reducer_input_dim=512,
                                                       local_position_embedding_dim=8,
                                                       encoder_flag=False,
                                                       monophonic_flag=False,
                                                       nade_flag=nade)
        else:
            processor_encoder = BachBeatsCPCDataProcessor(dataset=dataset,
                                                          embedding_dim=512 - 8,
                                                          reducer_input_dim=512,
                                                          local_position_embedding_dim=8,
                                                          encoder_flag=True,
                                                          monophonic_flag=False,
                                                          nade_flag=nade,
                                                          cpc_model=cpc_model)

            processor_decoder = BachBeatsCPCDataProcessor(dataset=dataset,
                                                          embedding_dim=512 - 8,
                                                          reducer_input_dim=512,
                                                          local_position_embedding_dim=8,
                                                          encoder_flag=False,
                                                          monophonic_flag=False,
                                                          nade_flag=nade,
                                                          cpc_model=cpc_model)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'bach_small':
        metadatas = [
            FermataMetadata(),
            TickMetadata(subdivision=subdivision),
            KeyMetadata()
        ]

        voices_ids = [0, 1, 2, 3]

        if cpc_config_name is not None:
            # notes to compute the first cpc code, we need to waste block_size tokens
            cpc_model = init_cpc_model(cpc_config_name)
            num_tokens_per_block = cpc_model.dataloader_generator.num_tokens_per_block // (subdivision * len(voices_ids))
            sequence_size += num_tokens_per_block

        chorale_dataset_kwargs = {
            'voice_ids': voices_ids,
            'metadatas': metadatas,
            'sequences_size': sequence_size,
            'subdivision': subdivision,
        }

        dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
            name='bach_chorales_beats_test',
            **chorale_dataset_kwargs
        )

        if cpc_config_name is None:
            processor_encoder = BachBeatsDataProcessor(dataset=dataset,
                                                       embedding_dim=512 - 8,
                                                       reducer_input_dim=512,
                                                       local_position_embedding_dim=8,
                                                       encoder_flag=True,
                                                       monophonic_flag=False,
                                                       nade_flag=nade)

            processor_decoder = BachBeatsDataProcessor(dataset=dataset,
                                                       embedding_dim=512 - 8,
                                                       reducer_input_dim=512,
                                                       local_position_embedding_dim=8,
                                                       encoder_flag=False,
                                                       monophonic_flag=False,
                                                       nade_flag=nade)
        else:
            processor_encoder = BachBeatsCPCDataProcessor(dataset=dataset,
                                                          embedding_dim=512 - 8,
                                                          reducer_input_dim=512,
                                                          local_position_embedding_dim=8,
                                                          encoder_flag=True,
                                                          monophonic_flag=False,
                                                          nade_flag=nade,
                                                          cpc_model=cpc_model)

            processor_decoder = BachBeatsCPCDataProcessor(dataset=dataset,
                                                          embedding_dim=512 - 8,
                                                          reducer_input_dim=512,
                                                          local_position_embedding_dim=8,
                                                          encoder_flag=False,
                                                          monophonic_flag=False,
                                                          nade_flag=nade,
                                                          cpc_model=cpc_model)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'lsdb':
        # leadsheet_dataset_kwargs = {
        #     'sequences_size': 24,
        # }
        # leadsheet_dataset_kwargs = {
        #     'sequences_size': 32,
        # }
        leadsheet_dataset_kwargs = {
            'sequences_size': 12,
        }
        dataset: LsdbDataset = dataset_manager.get_dataset(
            name='lsdb',
            **leadsheet_dataset_kwargs
        )
        processor_encoder = LsdbDataProcessor(dataset=dataset,
                                              embedding_dim=512 - 8,
                                              reducer_input_dim=512,
                                              local_position_embedding_dim=8)

        processor_decoder = LsdbDataProcessor(dataset=dataset,
                                              embedding_dim=512 - 8,
                                              reducer_input_dim=512,
                                              local_position_embedding_dim=8)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'velocity_quantization': velocity_quantization,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='orchestra',
                                                   block_attention=block_attention)

        processor_decoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='piano',
                                                   block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_large':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'velocity_quantization': velocity_quantization,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement_large',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='orchestra',
                                                   block_attention=block_attention)

        processor_decoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='piano',
                                                   block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_small':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'velocity_quantization': velocity_quantization,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='orchestra',
                                                   block_attention=block_attention)

        processor_decoder = ReductionDataProcessor(dataset=dataset,
                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                   reducer_input_dim=reducer_input_dim,
                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                   flag='piano',
                                                   block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'velocity_quantization': velocity_quantization,
            'max_transposition': max_transposition,
            'integrate_discretization': True,
            'alignement_type': 'complete',
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag='piano',
                                                     block_attention=block_attention,
                                                     nade=nade,
                                                     double_conditioning=double_conditioning)

        processor_decoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag='orchestra',
                                                     block_attention=block_attention,
                                                     nade=nade,
                                                     double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='instruments',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement_small':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'velocity_quantization': velocity_quantization,
            'max_transposition': max_transposition,
            'integrate_discretization': True,
            'alignement_type': 'complete',
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag='piano',
                                                     block_attention=block_attention,
                                                     nade=nade,
                                                     double_conditioning=double_conditioning)

        processor_decoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag='orchestra',
                                                     block_attention=block_attention,
                                                     nade=nade,
                                                     double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='instruments',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement_midiPiano':
        # For now just try a small value, anyway exception if too small
        mean_number_messages_per_time_frame = 14

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'mean_number_messages_per_time_frame': mean_number_messages_per_time_frame,
            'integrate_discretization': True,
            'alignement_type': 'complete',
        }
        dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
            name='arrangement_midiPiano',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='piano',
                                                              block_attention=block_attention,
                                                              nade=nade,
                                                              double_conditioning=double_conditioning)

        processor_decoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='orchestra',
                                                              block_attention=block_attention,
                                                              nade=nade,
                                                              double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                                   reducer_input_dim=reducer_input_dim,
                                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                                   flag='instruments',
                                                                   block_attention=block_attention,
                                                                   nade=nade,
                                                                   double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement_midiPiano_small':

        mean_number_messages_per_time_frame = 14

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'mean_number_messages_per_time_frame': mean_number_messages_per_time_frame,
            'integrate_discretization': True,
            'alignement_type': 'complete'
        }
        dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
            name='arrangement_midiPiano_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='piano',
                                                              block_attention=block_attention,
                                                              nade=nade,
                                                              double_conditioning=double_conditioning)

        processor_decoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='orchestra',
                                                              block_attention=block_attention,
                                                              nade=nade,
                                                              double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementMidiPianoDataProcessor(dataset=dataset,
                                                                   embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                                   reducer_input_dim=reducer_input_dim,
                                                                   local_position_embedding_dim=local_position_embedding_dim,
                                                                   flag='instruments',
                                                                   block_attention=block_attention,
                                                                   nade=nade,
                                                                   double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement_voice':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'integrate_discretization': True,
            'alignement_type': 'complete',
            'compute_statistics_flag': False,
        }
        dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
            name='arrangement_voice',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='piano',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        processor_decoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='orchestra',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                               embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                               reducer_input_dim=reducer_input_dim,
                                                               local_position_embedding_dim=local_position_embedding_dim,
                                                               flag='instruments',
                                                               block_attention=block_attention,
                                                               nade=nade,
                                                               double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'arrangement_voice_small':

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'integrate_discretization': True,
            'alignement_type': 'complete',
            'compute_statistics_flag': False,
        }
        dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
            name='arrangement_voice_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='piano',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        processor_decoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                          embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                          reducer_input_dim=reducer_input_dim,
                                                          local_position_embedding_dim=local_position_embedding_dim,
                                                          flag='orchestra',
                                                          block_attention=block_attention,
                                                          nade=nade,
                                                          double_conditioning=double_conditioning)

        processor_encodencoder = ArrangementVoiceDataProcessor(dataset=dataset,
                                                               embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                               reducer_input_dim=reducer_input_dim,
                                                               local_position_embedding_dim=local_position_embedding_dim,
                                                               flag='instruments',
                                                               block_attention=block_attention,
                                                               nade=nade,
                                                               double_conditioning=double_conditioning)

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    # elif dataset_type == 'arrangement_minimal':
    #
    #     arrangement_dataset_kwargs = {
    #         'transpose_to_sounding_pitch': True,
    #         'subdivision': subdivision,
    #         'sequence_size': sequence_size,
    #         'velocity_quantization': velocity_quantization,
    #         'max_transposition': max_transposition,
    #         'compute_statistics_flag': False
    #     }
    #     dataset: ArrangementDataset = dataset_manager.get_dataset(
    #         name='arrangement',
    #         **arrangement_dataset_kwargs
    #     )
    #
    #     reducer_input_dim = num_heads * per_head_dim
    #
    #     processor_encoder = ArrangementDataProcessorMinimal(dataset=dataset,
    #                                                         embedding_dim=reducer_input_dim - local_position_embedding_dim,
    #                                                         reducer_input_dim=reducer_input_dim,
    #                                                         local_position_embedding_dim=local_position_embedding_dim,
    #                                                         flag_orchestra=False,
    #                                                         block_attention=block_attention)
    #
    #     processor_decoder = ArrangementDataProcessorMinimal(dataset=dataset,
    #                                                         embedding_dim=reducer_input_dim - local_position_embedding_dim,
    #                                                         reducer_input_dim=reducer_input_dim,
    #                                                         local_position_embedding_dim=local_position_embedding_dim,
    #                                                         flag_orchestra=True,
    #                                                         block_attention=block_attention)
    #
    #     processor_encodencoder = None
    #
    #     return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'ar':
        dataset: ARDataset = ARDataset(
            phis=[0.9],
            length=128,
            c=0)

        # todo create BachTransformer and put BachBeats data processor in it
        processor_encoder = ARDataProcessor(dataset=dataset)

        processor_decoder = ARDataProcessor(dataset=dataset)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_categorical':
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'group_instrument_per_section': group_instrument_per_section
        }
        dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
            name='arrangement_categorical',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionCategoricalDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='orchestra',
                                                              block_attention=block_attention)

        processor_decoder = ReductionCategoricalDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='piano',
                                                              block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_categorical_small':

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'group_instrument_per_section': group_instrument_per_section
        }
        dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
            name='arrangement_categorical_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionCategoricalDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='orchestra',
                                                              block_attention=block_attention)

        processor_decoder = ReductionCategoricalDataProcessor(dataset=dataset,
                                                              embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                              reducer_input_dim=reducer_input_dim,
                                                              local_position_embedding_dim=local_position_embedding_dim,
                                                              flag='piano',
                                                              block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_midiPiano':
        # For now just try a small value, anyway exception if too small
        mean_number_messages_per_time_frame = 14

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'mean_number_messages_per_time_frame': mean_number_messages_per_time_frame,
            'integrate_discretization': True
        }
        dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
            name='arrangement_midiPiano',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionMidiPianoDataProcessor(dataset=dataset,
                                                            embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                            reducer_input_dim=reducer_input_dim,
                                                            local_position_embedding_dim=local_position_embedding_dim,
                                                            flag='orchestra',
                                                            block_attention=block_attention)

        processor_decoder = ReductionMidiPianoDataProcessor(dataset=dataset,
                                                            embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                            reducer_input_dim=reducer_input_dim,
                                                            local_position_embedding_dim=local_position_embedding_dim,
                                                            flag='piano',
                                                            block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder

    elif dataset_type == 'reduction_midiPiano_small':

        #  Todo: compuyte value before ?
        # For now just try a small value, anyway exception if too small
        mean_number_messages_per_time_frame = 14

        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': subdivision,
            'sequence_size': sequence_size,
            'max_transposition': max_transposition,
            'compute_statistics_flag': False,
            'mean_number_messages_per_time_frame': mean_number_messages_per_time_frame,
            'integrate_discretization': True
        }
        dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
            name='arrangement_midiPiano_small',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ReductionMidiPianoDataProcessor(dataset=dataset,
                                                            embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                            reducer_input_dim=reducer_input_dim,
                                                            local_position_embedding_dim=local_position_embedding_dim,
                                                            flag='orchestra',
                                                            block_attention=block_attention)

        processor_decoder = ReductionMidiPianoDataProcessor(dataset=dataset,
                                                            embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                            reducer_input_dim=reducer_input_dim,
                                                            local_position_embedding_dim=local_position_embedding_dim,
                                                            flag='piano',
                                                            block_attention=block_attention)

        processor_encodencoder = None

        return dataset, processor_decoder, processor_encoder, processor_encodencoder
    else:
        raise NotImplementedError
