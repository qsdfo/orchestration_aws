import itertools

import torch
import click
from DatasetManager.dataset_manager import DatasetManager

import dataset_import
from Transformer.constants import MARIO_MELODY
from Transformer.config import get_config
from Transformer.generate import generation_from_file, reduction_from_file, generation_bach, generation_bach_nade
from Transformer.transformer import Transformer
from Transformer.visualise import visualize_arrangement


@click.command()
# Model (high-level)
@click.option('--block_attention', is_flag=True,
              help='Do we use block attention ?')
@click.option('--hierarchical', is_flag=True,
              help='Do we connect encoder and decoder in a hierarchical way')
@click.option('--nade', is_flag=True,
              help='Orderless auto-regressive model')
@click.option('--conditioning', is_flag=True,
              help='condition on set of constraints')
@click.option('--double_conditioning', type=click.Choice(['concatenate', 'stack_conditioning_layer',
                                                          'condition_encoder']),
              default=None, help='Second conditioning information (like instruments presence for arrangement)')
@click.option('--instrument_presence_in_encoder', is_flag=True,
              help='if activated, the instrument presence vector is appended in the encoder')
@click.option('--cpc_config_name', type=str, default=None,
              help='path to cpc model and config. If None, cpc is not used')
# Model (low-level)
@click.option('--num_layers', default=6,
              help='number of layers of the LSTMs')
@click.option('--dropout', default=0.1,
              help='amount of dropout between layers')
@click.option('--input_dropout', default=0.,
              help='amount of dropout on input')
@click.option('--input_dropout_token', default=0.,
              help='amount of dropout on input tokens (not embeddings)')
@click.option('--per_head_dim', default=64,
              help='Feature dimension in each head')
@click.option('--num_heads', default=8,
              help='Number of heads')
@click.option('--local_position_embedding_dim', default=8,
              help='Embedding size for local positions')
@click.option('--position_ff_dim', default=2048,
              help='Hidden dimension of the position-wise ffnn')
@click.option('--enc_dec_conditioning', default='split', type=click.Choice(['split', 'single']),
              help='Mechanism for conditioning the decoder with the encoder')
# Learning
@click.option('--lr', type=float, default=1e-4,
              help='Learning rate')
@click.option('--batch_size', default=128,
              help='training batch size')
@click.option('--num_batches', default=None, type=int,
              help='Number of batches per epoch, None for all dataset')
@click.option('--num_epochs', default=2000,
              help='number of training epochs')
@click.option('--action', type=click.Choice(['train', 'train_from_checkpoint',
                                             'generate', 'generate_overfit',
                                             'visualize', 'visualize_overfit']),
              help='Choose the action to perform (train, generate, visualize...)')
@click.option('--loss_on_last_frame', is_flag=True,
              help='Compute loss only on o(t), and not on reconstruction of previous frames')
@click.option('--mixup', is_flag=True,
              help='Mixup manifold')
@click.option('--label_smoothing', default=0.001, type=float,
              help='Label smoothing')
@click.option('--scheduled_training', default=0, type=int,
              help='Scheduler for the learning rate. '
                   'The value indicates the number of warmup steps'
                   'in iteration over the whole dataset')
# Dataset
@click.option('--dataset_type',
              type=click.Choice(
                  ['bach', 'bach_small',
                   'lsdb',
                   'arrangement', 'arrangement_small', 'arrangement_minimal',
                   'arrangement_voice', 'arrangement_voice_small',
                   'reduction', 'reduction_large', 'reduction_small',
                   'ar',
                   'reduction_categorical', 'reduction_categorical_small',
                   'reduction_midiPiano', 'reduction_midiPiano_small',
                   'arrangement_midiPiano', 'arrangement_midiPiano_small']))
@click.option('--subdivision', default=4, type=int,
              help='subdivisions of qaurter note in dataset')
@click.option('--sequence_size', default=3, type=int,
              help='length of piano chunks')
@click.option('--velocity_quantization', default=2, type=int,
              help='number of possible velocities')
@click.option('--max_transposition', default=12, type=int,
              help='maximum pitch shift allowed when transposing for data augmentation')
@click.option('--group_instrument_per_section', is_flag=True,
              help='group instruments per section')
# Generation
@click.option('--midi_input', default=None, type=str,
              help='name of the midi input file to orchestrate')
@click.option('--temperature', default=1.2,
              help='Temperature for sampling')
@click.option('--num_examples_sampled', default=3,
              help='number of orchestration generated per given piano input')
@click.option('--suffix', default="", type=str,
              help='suffix for model name')
def main(block_attention,
         hierarchical,
         nade,
         num_layers,
         dropout,
         input_dropout,
         input_dropout_token,
         per_head_dim,
         num_heads,
         local_position_embedding_dim,
         position_ff_dim,
         enc_dec_conditioning,
         lr,
         batch_size,
         num_epochs,
         action,
         loss_on_last_frame,
         mixup,
         midi_input,
         temperature,
         num_batches,
         label_smoothing,
         scheduled_training,
         dataset_type,
         conditioning,
         double_conditioning,
         instrument_presence_in_encoder,
         cpc_config_name,
         num_examples_sampled,
         suffix,
         subdivision,
         sequence_size,
         velocity_quantization,
         max_transposition,
         group_instrument_per_section
         ):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(gpu_ids)

    config = get_config()

    num_layers_l = [2, 3, 4, 5, 6]
    enc_dec_conditionings_l = ['split', 'single']
    sequence_sizes_l = [3, 5, 7]
    grid_search = False
    if grid_search:
        configs = list(itertools.product(*[num_layers_l, enc_dec_conditionings_l, sequence_sizes_l]))
        write_suffix = True
    else:
        configs = [(num_layers, enc_dec_conditioning, sequence_size)]
        write_suffix = False

    for this_config in configs:
        num_layers, enc_dec_conditioning, sequence_size = this_config
        if write_suffix:
            this_suffix = f'{suffix}_{num_layers}_{enc_dec_conditioning}_{sequence_size}'
        else:
            this_suffix = suffix

        # Get dataset
        dataset_manager = DatasetManager()
        dataset, processor_decoder, processor_encoder, processor_encodencoder = \
            dataset_import.get_dataset(dataset_manager, dataset_type, subdivision, sequence_size, velocity_quantization,
                                       max_transposition,
                                       num_heads, per_head_dim, local_position_embedding_dim, block_attention,
                                       group_instrument_per_section, nade, cpc_config_name, double_conditioning,
                                       instrument_presence_in_encoder)

        reduction_flag = dataset_type in ['reduction', 'reduction_small', 'reduction_large',
                                          'reduction_categorical', 'reduction_categorical_small',
                                          'reduction_midiPiano', 'reduction_midiPiano_small']

        if not conditioning:
            print("NO CONDITIONING ????!!!!!!!!!!!!")

        model = Transformer(dataset=dataset,
                            data_processor_encodencoder=processor_encodencoder,
                            data_processor_encoder=processor_encoder,
                            data_processor_decoder=processor_decoder,
                            num_heads=num_heads,
                            per_head_dim=per_head_dim,
                            position_ff_dim=position_ff_dim,
                            enc_dec_conditioning=enc_dec_conditioning,
                            hierarchical_encoding=hierarchical,
                            block_attention=block_attention,
                            nade=nade,
                            conditioning=conditioning,
                            double_conditioning=double_conditioning,
                            num_layers=num_layers,
                            dropout=dropout,
                            input_dropout=input_dropout,
                            input_dropout_token=input_dropout_token,
                            lr=lr, reduction_flag=reduction_flag,
                            gpu_ids=gpu_ids,
                            suffix=this_suffix,
                            mixup=mixup,
                            scheduled_training=scheduled_training
                            )

        if action in ['generate', 'visualize']:
            model.load()
            overfit_flag = False
        elif action in ['generate_overfit', 'train_from_checkpoint', 'visualize_overfit']:
            model.load_overfit()
            overfit_flag = True

        model.cuda()

        if action in ['train', 'train_from_checkpoint']:
            print(f"Train the model on gpus {gpu_ids}")
            model.train_model(cache_dir=dataset_manager.cache_dir,
                              batch_size=batch_size,
                              num_epochs=num_epochs,
                              num_batches=num_batches,
                              label_smoothing=label_smoothing,
                              loss_on_last_frame=loss_on_last_frame)
            overfit_flag = True

        if action in ['generate', 'generate_overfit']:
            print('Generation')
            ascii_melody = MARIO_MELODY
            # score, tensor_chorale, tensor_metadata = mode.generation_from_ascii(
            #     ascii_melody=ascii_melody
            # )
            # score.show()
            # score, tensor_chorale, tensor_metadata = model.generation(
            #     num_tokens_per_beat=8,
            #     num_beats=64 * 4,
            #     temperature=1.
            # )
            # score, tensor_chorale, tensor_metadata = model.generation(
            #     num_tokens_per_beat=8,
            #     num_beats=64 * 4,
            #     num_experiments=4,
            #     link_experiments=False,
            #     temperature=1.2
            # )
            # score, tensor_chorale, tensor_metadata = model.plot_attentions()
            # score, tensor_chorale, tensor_metadata = model.unconstrained_generation(
            #     num_tokens_per_beat=8,
            #     num_beats=64 * 4)

            if dataset_type in ['arrangement', 'arrangement_small',
                                'arrangement_midiPiano', 'arrangement_midiPiano_small',
                                'arrangement_voice', 'arrangement_voice_small']:
                # (oppposite to standard) increasing temperature reduce agitation
                # Cold means all event will eventually have almost same proba
                # Hot accentuates spikes

                # Number of complete pass over all time frames in, non auto-regressive sampling schemes
                number_sampling_steps = 1
                #  Allows to override dataset quantization for generation
                subdivision_generation = subdivision
                # banned_instruments = ["Violin_1", "Violin_2", "Violoncello", "Contrabass", "Viola"]
                banned_instruments = []
                # Used for instruments_presence model
                unknown_instruments = []
                source_folder = f"{config['datapath']}/source_for_generation/"
                sources = [
                    {"source_path": source_folder + "mouss_tableaux_small.xml",
                     "writing_name": "mouss_tableaux_small",
                     "writing_tempo": "adagio",
                     },
                    # {"source_path": source_folder + "guillaume_1.mid",
                    #  "writing_name": "guillaume_1",
                    #  "writing_tempo": "adagio"
                    #  },
                    # {"source_path": source_folder + "guillaume_2.xml",
                    #  "writing_name": "guillaume_2",
                    #  "writing_tempo": "adagio"
                    #  },
                    {"source_path": source_folder + "chopin_Prel_Op28_20.xml",
                     "writing_name": "chopin_Prel_Op28_20",
                     "writing_tempo": "largo"
                     },
                    {"source_path": source_folder + "b_1_1.xml",
                     "writing_name": "b_1_1",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "b_3_3.xml",
                     "writing_name": "b_3_3",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "b_3_4.xml",
                     "writing_name": "b_3_4",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "b_7_2.xml",
                     "writing_name": "b_7_2",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "testpiano.xml",
                     "writing_name": "testpiano",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "schubert_21_1.xml",
                     "writing_name": "schubert_21_1",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "schubert_20_1.xml",
                     "writing_name": "schubert_20_1",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "Mozart_Nachtmusik.xml",
                     "writing_name": "Mozart_Nachtmusik",
                     "writing_tempo": "adagio"
                     },
                ]
                if overfit_flag:
                    write_dir = model.log_dir_overfitted
                else:
                    write_dir = model.log_dir

                if midi_input is not None:
                    sources = [
                        {
                            'source_path': f'midi_inputs/{midi_input}',
                            'writing_name': f'{midi_input}',
                            'writing_tempo': 'adagio'
                        }
                    ]
                    write_dir = 'midi_inputs'

                for source in sources:
                    generation_from_file(
                        model=model,
                        temperature=temperature,
                        batch_size=num_examples_sampled,
                        filepath=source["source_path"],
                        write_dir=write_dir,
                        write_name=source["writing_name"],
                        banned_instruments=banned_instruments,
                        unknown_instruments=unknown_instruments,
                        writing_tempo=source["writing_tempo"],
                        subdivision=subdivision_generation,
                        number_sampling_steps=number_sampling_steps
                    )

            elif dataset_type in ['reduction', 'reduction_large', 'reduction_small',
                                  'reduction_categorical', 'reduction_categorical_small']:
                #  Allows to override dataset quantization for generation
                subdivision_generation = 8
                source_folder = f"{config['datapath']}/source_for_generation/"
                sources = [
                    {"source_path": source_folder + "b_7_2_orch.xml",
                     "writing_name": "b_7_2_orch",
                     "writing_tempo": "adagio"
                     },
                    # {"source_path": source_folder + "mouss_tableaux_orch.xml",
                    #  "writing_name": "mouss_tableaux_orch",
                    #  "writing_tempo": "adagio"
                    #  },
                    # {"source_path": source_folder + "Debussy_SuiteBergam_Passepied_orch.xml",
                    #  "writing_name": "Debussy_SuiteBergam_Passepied_orch",
                    #  "writing_tempo": "adagio"
                    #  },
                    # {
                    #     "source_path": source_folder + "Romantic Concert Piece for Brass Orchestra_orch.xml",
                    #     "writing_name": "Romantic Concert Piece for Brass Orchestra_orch",
                    #     "writing_tempo": "adagio"
                    #  },
                    # {
                    #     "source_path": source_folder + "mozart_25_1.xml",
                    #     "writing_name": "mozart_25_1",
                    #     "writing_tempo": "adagio"
                    # },
                    # {
                    #     "source_path": source_folder + "mozart_25_2.xml",
                    #     "writing_name": "mozart_25_2",
                    #     "writing_tempo": "adagio"
                    # },
                    # {
                    #     "source_path": source_folder + "mozart_25_3.xml",
                    #     "writing_name": "mozart_25_3",
                    #     "writing_tempo": "adagio"
                    # },
                    {"source_path": source_folder + "brahms_symphony_2_1.xml",
                     "writing_name": "brahms_symphony_2_1",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "haydn_symphony_91_1.xml",
                     "writing_name": "haydn_symphony_91_1",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "mozart_symphony_183_4.xml",
                     "writing_name": "mozart_symphony_183_4",
                     "writing_tempo": "adagio"
                     },
                    {"source_path": source_folder + "mozart_symphony_183_2.xml",
                     "writing_name": "mozart_symphony_183_2",
                     "writing_tempo": "adagio"
                     },
                ]
                for source in sources:
                    reduction_from_file(
                        model=model,
                        temperature=temperature,
                        batch_size=num_examples_sampled,
                        filepath=source["source_path"],
                        write_name=source["writing_name"],
                        overfit_flag=overfit_flag,
                        writing_tempo=source["writing_tempo"],
                        subdivision=subdivision_generation
                    )

            elif dataset_type == "lsdb":
                score, tensor_chorale, tensor_metadata = model.generation()
                score.write('xml', 'results/test.xml')

            elif dataset_type in ['bach', 'bach_small']:
                if nade and (not conditioning):
                    scores = generation_bach_nade(model=model,
                                                  temperature=temperature,
                                                  ascii_melody=ascii_melody,
                                                  batch_size=num_examples_sampled,
                                                  force_melody=False, )
                else:
                    scores = generation_bach(model=model,
                                             temperature=temperature,
                                             ascii_melody=ascii_melody,
                                             batch_size=num_examples_sampled,
                                             force_melody=False)

                if overfit_flag:
                    writing_dir = model.log_dir_overfitted
                else:
                    writing_dir = model.log_dir

                for batch_index, score in enumerate(scores):
                    score.write('xml', f'{writing_dir}/{batch_index}.xml')
        elif action in ['visualize', 'visualize_overfit']:
            log_dir = model.log_dir if action == 'visualize' else model.log_dir_overfitted
            visualize_arrangement(model, batch_size, log_dir)
    return


if __name__ == '__main__':
    main()
