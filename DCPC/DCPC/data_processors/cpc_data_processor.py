from torch import nn

from DCPC.dataloaders.cpc_dataloader import CPCDataloaderGenerator


class CPCDataProcessor(nn.Module):
    """
    Abstract class to embed blocks
    """

    def __init__(self, dataloader_generator, embedding_size):
        super(CPCDataProcessor, self).__init__()
        self.reference_dataset = (dataloader_generator.dataset
                                  if not (
                isinstance(dataloader_generator.dataset, list)
                or isinstance(dataloader_generator.dataset, tuple))
                                  else dataloader_generator.dataset[0])
        self.dataloader_generator: CPCDataloaderGenerator = dataloader_generator
        self.embedding_size = embedding_size

    def embed_block(self, block):
        """

        :param block: (..., num_tokens_per_block)
        :return: (..., num_tokens_per_block, embedding_size)
        """
        raise NotImplementedError

    def embed(self, tensor_dict):
        """
        to be called after preprocess

        :param tensor_dict: dict of tensors of shape (... num_blocks, num_tokens_per_block)
        :return:
        """
        return {
            k: self.embed_block(v)
            for k, v in tensor_dict.items()
        }

    def preprocess(self, tensor_dict):
        """
        put to cuda and format as (... num_blocks, num_tokens_per_block)
        :param tensor_dict:
        :return:
        """
        return {
            k: self.cut_blocks(v.long())
            for k, v in tensor_dict.items()
        }

    def cut_blocks(self, x):
        """

        :param x: (batch_size, dataset dependent shape)
        :return: (batch_size, num_blocks, num_tokens_per_block)
        """
        raise NotImplementedError

    def forward(self, x):
        return self.embed_block(x)
