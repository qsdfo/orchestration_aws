import torch
import click
from DatasetManager.dataset_manager import DatasetManager
import dataset_import
from Transformer.generate import generation_from_file
from Transformer.transformer import Transformer


@click.command()
@click.option('--midi_input', default=None, type=str,
              help='name of the midi input file to orchestrate')
@click.option('--temperature', default=1.2,
              help='Temperature for sampling')
@click.option('--num_examples_sampled', default=3,
              help='number of orchestration generated per given piano input')
@click.option('--suffix', default="", type=str,
              help='suffix for model name')
def main(midi_input,
         temperature,
         num_examples_sampled,
         suffix,
         ):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(gpu_ids)

    hierarchical = False
    nade = False
    num_layers = 6
    dropout = 0.
    input_dropout = 0.
    input_dropout_token = 0.
    per_head_dim = 64
    num_heads = 8
    local_position_embedding_dim = 8
    position_ff_dim = 2048
    enc_dec_conditioning = 'split'
    lr = 1
    mixup = None
    scheduled_training = 0
    dataset_type = 'arrangement_voice'
    conditioning = True
    double_conditioning = None
    subdivision = 16
    sequence_size = 7
    velocity_quantization = 2
    max_transposition = 12
    group_instrument_per_section = False
    reduction_flag = False
    instrument_presence_in_encoder = False
    cpc_config_name = None
    block_attention = False

    # Get dataset
    dataset_manager = DatasetManager()
    dataset, processor_decoder, processor_encoder, processor_encodencoder = \
        dataset_import.get_dataset(dataset_manager, dataset_type, subdivision, sequence_size, velocity_quantization,
                                   max_transposition,
                                   num_heads, per_head_dim, local_position_embedding_dim, block_attention,
                                   group_instrument_per_section, nade, cpc_config_name, double_conditioning,
                                   instrument_presence_in_encoder)

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
                        suffix=suffix,
                        mixup=mixup,
                        scheduled_training=scheduled_training
                        )

    model.load_overfit()
    model.cuda()

    print('Generation')
    #  Allows to override dataset quantization for generation
    subdivision_generation = subdivision

    source = {
            'source_path': f'midi_inputs/{midi_input}',
            'writing_name': f'{midi_input}',
            'writing_tempo': 'adagio'
        }

    write_dir = 'midi_inputs'

    generation_from_file(
        model=model,
        temperature=temperature,
        batch_size=num_examples_sampled,
        filepath=source["source_path"],
        write_dir=write_dir,
        write_name=source["writing_name"],
        banned_instruments=[],
        unknown_instruments=[],
        writing_tempo=source["writing_tempo"],
        subdivision=subdivision_generation,
        number_sampling_steps=1
    )
    return


if __name__ == '__main__':
    main()
