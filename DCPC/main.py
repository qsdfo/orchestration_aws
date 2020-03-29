import importlib
import os
import shutil
from datetime import datetime

import click
import torch

from DCPC.cpc_model.cpc import CPC
from DCPC.data_helpers import get_cpc_dataloader_generators, get_cpc_dataprocessor, \
    get_vector_quantizer


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-c', '--config', type=click.Path(exists=True))
def main(train,
         load,
         config):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        config['timestamp'] = timestamp

    if load:
        model_path = os.path.dirname(config_path)
    else:
        model_path = f'models/{config["savename"]}_{timestamp}'

    # assert something?
    # assert not (os.path.exists(model_path) and train)

    # Load dataset and data_processor
    dataloader_generator = get_cpc_dataloader_generators(
        dataset_type=config['dataset_type'],
        **config['dataloader_kwargs']
    )

    dataprocessor = get_cpc_dataprocessor(
        dataloader_generator=dataloader_generator,
        embedding_size=config['embedding_size']
    )

    vector_quantizer = get_vector_quantizer(
        vector_quantizer_type=config['vector_quantizer_type'],
        vector_quantizer_kwargs=config['vector_quantizer_kwargs'],
        output_dim=config['encoding_size_zt'],
        initialize=not load
    )

    #  Instantiate model
    model = CPC(
        # loader
        dataloader_generator=dataloader_generator,
        # modules
        dataprocessor=dataprocessor,
        vector_quantizer=vector_quantizer,

        # TODO put encoder in modules
        embedding_size=config['embedding_size'],
        # enc
        num_layers_enc=config['num_layers_enc'],
        encoding_size_zt=config['encoding_size_zt'],
        bidirectional_enc=config['bidirectional_enc'],
        # ar
        num_layers_ar=config['num_layers_ar'],
        encoding_size_ct=config['encoding_size_ct'],
        bidirectional_ar=config['bidirectional_ar'],
        rnn_hidden_size=config['rnn_hidden_size'],
        # general
        dropout=config['dropout'],
        lr=config['lr'],
        beta=config['beta'],
        corrput_labels=config['corrupt_labels'],
        model_path=model_path
    )

    # Load model
    if load:
        model.load()

    model.cuda()

    if train:
        # Copy .py config file in the save directory before training
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        shutil.copy(config_path, f'{model_path}/config.py')

        print(f'Train the model on gpus {gpu_ids}')
        model.train_model(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_epochs=config['num_epochs'],
            num_negative_samples=config['num_negative_samples']
        )

    print(f'Visualize examples grouped by maximal activations')

    model.visualize_discrete_encoding_bach()
    # TODO?
    # model.visualize_keys_bach(config)


if __name__ == '__main__':
    main()
