from DatasetManager.chorale_dataset import ChoraleDataset
from DCPC.cpc_model.vector_quantizer import ProductVectorQuantizer, NoQuantization
from DCPC.data_processors.bach_cpc_data_processor import BachCPCDataProcessor
from DCPC.dataloaders.bach_cpc_dataloader import BachCPCDataloaderGenerator, BachCPCSmallDataloaderGenerator


def get_cpc_dataloader_generators(
        dataset_type,
        num_tokens_per_block,
        num_blocks_left,
        num_blocks_right,
        negative_sampling_method):
    if dataset_type.lower() == 'bach':
        return BachCPCDataloaderGenerator(
            num_tokens_per_block=num_tokens_per_block,
            num_blocks_left=num_blocks_left,
            num_blocks_right=num_blocks_right,
            negative_sampling_method=negative_sampling_method
        )
    if dataset_type.lower() == 'bach_small':
        return BachCPCSmallDataloaderGenerator(
            num_tokens_per_block=num_tokens_per_block,
            num_blocks_left=num_blocks_left,
            num_blocks_right=num_blocks_right,
            negative_sampling_method=negative_sampling_method
        )
    else:
        raise NotImplementedError


def get_cpc_dataprocessor(
        dataloader_generator,
        embedding_size
):
    if isinstance(dataloader_generator.dataset, ChoraleDataset):
        return BachCPCDataProcessor(dataloader_generator, embedding_size)
    elif all([isinstance(dataset, ChoraleDataset)
              for dataset in dataloader_generator.dataset]):
        return BachCPCDataProcessor(dataloader_generator, embedding_size)
    else:
        raise NotImplementedError


def get_vector_quantizer(vector_quantizer_type,
                         vector_quantizer_kwargs,
                         output_dim,
                         initialize):
    if vector_quantizer_type == 'product':
        return ProductVectorQuantizer(
            output_dim=output_dim,
            initialize=initialize,
            **vector_quantizer_kwargs
        )
    elif vector_quantizer_type == 'none':
        return NoQuantization()
    else:
        raise NotImplementedError


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        if type(value) == list:
            print(f'{key.capitalize()}: [%s]' % ', '.join(map(str, value)))
        else:
            print(f'{key.capitalize()}: {value:.6}', end=endstr)
