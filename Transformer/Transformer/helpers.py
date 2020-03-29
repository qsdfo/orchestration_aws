import importlib
import os
import shutil

import torch
import DCPC
from DCPC.cpc_model.cpc import CPC
from DCPC.data_helpers import get_vector_quantizer, get_cpc_dataprocessor, get_cpc_dataloader_generators


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        print(f'{key.capitalize()}: {value:.6}', end=endstr)


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return tensor.to('cuda')
    else:
        return tensor

def to_numpy(variable):
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def number_weights(model):
    counter = 0
    for parameter in model.parameters():
        counter += parameter.view(-1).shape[0]
    return counter


def init_cpc_model(cpc_config_name):
    # Init CPC model
    # Copy config file locally
    path_to_DCPC = os.path.abspath(DCPC.__path__[0])
    copy_path = f'cpc_configs/{cpc_config_name}.py'
    shutil.copyfile(f'{path_to_DCPC}/../models/{cpc_config_name}/config.py', copy_path)
    config_module_name = os.path.splitext(copy_path)[0].replace('/', '.')
    config_cpc = importlib.import_module(config_module_name).config
    model_path = f'{path_to_DCPC}/../models/{cpc_config_name}'

    # Load dataset and data_processor
    dataloader_generator = get_cpc_dataloader_generators(
        dataset_type=config_cpc['dataset_type'],
        **config_cpc['dataloader_kwargs']
    )
    dataprocessor = get_cpc_dataprocessor(
        dataloader_generator=dataloader_generator,
        embedding_size=config_cpc['embedding_size']
    )
    vector_quantizer = get_vector_quantizer(
        vector_quantizer_type=config_cpc['vector_quantizer_type'],
        vector_quantizer_kwargs=config_cpc['vector_quantizer_kwargs'],
        output_dim=config_cpc['encoding_size_zt'],
        initialize=False
    )
    #  Instantiate model
    model = CPC(
        # loader
        dataloader_generator=dataloader_generator,
        # modules
        dataprocessor=dataprocessor,
        vector_quantizer=vector_quantizer,
        embedding_size=config_cpc['embedding_size'],
        # enc
        num_layers_enc=config_cpc['num_layers_enc'],
        encoding_size_zt=config_cpc['encoding_size_zt'],
        bidirectional_enc=config_cpc['bidirectional_enc'],
        # ar
        num_layers_ar=config_cpc['num_layers_ar'],
        encoding_size_ct=config_cpc['encoding_size_ct'],
        bidirectional_ar=config_cpc['bidirectional_ar'],
        rnn_hidden_size=config_cpc['rnn_hidden_size'],
        # general
        dropout=config_cpc['dropout'],
        lr=config_cpc['lr'],
        beta=config_cpc['beta'],
        corrput_labels=config_cpc['corrupt_labels'],
        model_path=model_path
    )
    # Load model, freeze and cuda
    model.load()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.cuda()
    return model
