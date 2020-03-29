from DatasetManager.chorale_dataset import ChoraleDataset

from DCPC.data_processors.cpc_data_processor import CPCDataProcessor
import torch
from torch import nn
import numpy as np


class BachCPCDataProcessor(CPCDataProcessor):
    """
    Abstract class to embed blocks
    """

    def __init__(self, dataloader_generator, embedding_size):
        super(BachCPCDataProcessor, self).__init__(
            dataloader_generator=dataloader_generator,
            embedding_size=embedding_size)

        assert isinstance(self.reference_dataset, ChoraleDataset)

        #  Embedding per voice
        self.num_notes_per_voice = [len(d) for d in self.reference_dataset.note2index_dicts]
        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + 1, self.embedding_size)
                for num_embeddings in self.num_notes_per_voice
            ]
        )

    def embed_block(self, block):
        """

        :param block: (..., num_tokens_per_block)
        :return: (..., num_tokens_per_block, embedding_size)
        """
        batch_dims = block.size()[:-1]
        product_batch_dim = np.prod(batch_dims)
        num_tokens_per_block = block.size()[-1]
        block = block.view(-1, num_tokens_per_block)

        # must reshape using num_voices...
        num_voices = self.reference_dataset.num_voices

        block = block.view(product_batch_dim, -1, num_voices)

        separate_voices = block.split(split_size=1, dim=2)
        separate_voices = [
            embedding(voice[:, :, 0])[:, :, None, :]
            for voice, embedding
            in zip(separate_voices, self.note_embeddings)
        ]
        x = torch.cat(separate_voices, 2)
        # TODO check with measures

        x = x.view(product_batch_dim, -1, self.embedding_size)
        x = x.view(*batch_dims, num_tokens_per_block, self.embedding_size)
        return x

    def cut_blocks(self, x):
        """

        :param x: (..., num_voices, num_ticks) of appropriate dimensions
        :return: (..., num_blocks, num_tokens_per_block)
        """
        num_voices, num_ticks = x.size()[-2:]
        remaining_dims = x.size()[:-2]

        x = x.view(-1, num_voices, num_ticks).contiguous()
        x = x.permute(0, 2, 1).contiguous().view(-1, num_voices * num_ticks)

        num_tokens_per_block = self.dataloader_generator.num_tokens_per_block
        assert x.size(1) % num_tokens_per_block == 0
        x = x.split(num_tokens_per_block, dim=1)
        x = torch.cat(
            [t.unsqueeze(1) for t in x], dim=1
        )

        num_blocks = x.size(1)
        x = x.view(*remaining_dims, num_blocks, num_tokens_per_block)
        return x
