import torch
from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DCPC.dataloaders.cpc_dataloader import CPCDataloaderGenerator

subdivision = 4
num_voices = 4
metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=subdivision),
    KeyMetadata()
]


class BachCPCDataloaderGenerator(CPCDataloaderGenerator):
    def __init__(self,
                 num_tokens_per_block,
                 num_blocks_left,
                 num_blocks_right,
                 negative_sampling_method,
                 *args, **kwargs):
        """

        :param num_tokens_per_block:
        :param num_blocks_left:
        :param num_blocks_right:
        :param num_negative_samples:
        :param negative_sampling_method:
        :param args:
        :param kwargs:
        """
        assert num_tokens_per_block % (subdivision * num_voices) == 0
        super(BachCPCDataloaderGenerator, self).__init__(
            num_tokens_per_block,
            num_blocks_left,
            num_blocks_right,
            negative_sampling_method)
        # load dataset
        self.dataset = self._dataset()

    def _dataset(self):
        """
        Loads the appropriate dataset depending on the sampling method
        :return: ChoraleDataset or tuple(ChoraleDataset)
        """

        dataset_manager = DatasetManager()
        self.cache_dir = dataset_manager.cache_dir

        if self.negative_sampling_method == 'random_bad':
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            chorale_dataset_positive_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_positive,
                'subdivision':    subdivision,
            }

            dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats',
                **chorale_dataset_positive_kwargs
            )

            return dataset
        elif self.negative_sampling_method == 'same_sequence':
            # FIXME for the moment, exactly the same as 'random' _dataset
            dataset_manager = DatasetManager()
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            chorale_dataset_positive_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_positive,
                'subdivision':    subdivision,
            }

            dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats',
                **chorale_dataset_positive_kwargs
            )

            return dataset
        if self.negative_sampling_method == 'random':
            dataset_manager = DatasetManager()
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            chorale_dataset_positive_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_positive,
                'subdivision':    subdivision,
            }

            dataset_positive: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats',
                **chorale_dataset_positive_kwargs
            )
            num_tokens_per_beat = subdivision * num_voices
            num_beats_negative = self.num_tokens_per_block // num_tokens_per_beat
            chorale_dataset_negative_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_negative,
                'subdivision':    subdivision,
            }

            dataset_negative: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats',
                **chorale_dataset_negative_kwargs
            )
            return dataset_positive, dataset_negative
        else:
            raise NotImplementedError

    def dataloader(self,
                   batch_size,
                   num_negative_samples
                   ):
        """

        :return: torch Dataloader, returns a dict of
        {
        'x_left': (batch_size, num_blocks_left, num_tokens_per_block)
        'x_right': (batch_size, num_blocks_right, num_tokens_per_block)
        'negative_samples': (batch_size, num_negative_samples, num_blocks_right,
        num_tokens_per_block)
        }

        """
        #
        if self.negative_sampling_method == 'random_bad':
            return self._dataloader_random_bad(batch_size=batch_size,
                                               num_negative_samples=num_negative_samples)
        elif self.negative_sampling_method == 'random':
            return self._dataloader_random(batch_size=batch_size,
                                           num_negative_samples=num_negative_samples)
        elif self.negative_sampling_method == 'same_sequence':
            return self._dataloader_same_sequence(batch_size=batch_size,
                                                  num_negative_samples=None)
        elif self.negative_sampling_method == 'shuffle':
            raise NotImplementedError

        else:
            raise NotImplementedError

    def block_dataloader(self,
                         batch_size):
        """

            :return: torch Dataloader, returns batches of
            (batch_size, num_blocks=1, num_tokens_per_block)
            }

        """
        dataset_manager = DatasetManager()
        num_tokens_per_beat = subdivision * num_voices

        # Positive dataset
        num_beats = self.num_tokens_per_block // num_tokens_per_beat
        chorale_dataset_kwargs = {
            'voice_ids':      [0, 1, 2, 3],
            'metadatas':      metadatas,
            'sequences_size': num_beats,
            'subdivision':    subdivision,
        }

        dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
            name='bach_chorales_beats',
            **chorale_dataset_kwargs
        )
        return [({'x': t[0]} # discard metadata
                 for t in dataloader)
                for dataloader
                in dataset.data_loaders(batch_size)]

    def _dataloader_same_sequence(self, batch_size, num_negative_samples):
        """
        Dataloader for negative_sampling_method == 'random'
        :param batch_size:
        :return:
        """
        # dataset should be initialized by self._dataset
        # WARNING num_negative_samples parameter is not used
        num_negative_samples = self.num_blocks_right + self.num_blocks_left - 1
        assert self.dataset is not None
        num_tokens_left = self.num_tokens_per_block * self.num_blocks_left
        num_tokens_per_beat = subdivision * num_voices

        dataloaders = self.dataset.data_loaders(
            batch_size=batch_size,
            cache_dir=self.cache_dir
        )

        # Generate dataloaders
        def _aggregate_dataloader(dataloader):
            for p in dataloader:
                # remove metadata
                p = p[0]

                x_left = p[:, :, :num_tokens_left // num_voices]
                x_right = p[:, :, num_tokens_left // num_voices:]

                # generate negative samples
                negative_sample = []
                for k in range(self.num_blocks_right):
                    x_right_split = x_right.unsqueeze(1).split(self.num_tokens_per_block //
                                                               num_voices,
                                                               dim=3)
                    neg_k = torch.cat(
                        (*x_left.unsqueeze(1).split(self.num_tokens_per_block // num_voices, dim=3),
                         *x_right_split[:k],
                         *x_right_split[k + 1:]),
                        dim=1
                    ).unsqueeze(2)
                    negative_sample.append(neg_k)
                negative_sample = torch.cat(negative_sample,
                                            dim=2)

                negative_sample = negative_sample.view(
                    batch_size,
                    num_negative_samples,
                    self.num_blocks_right,
                    num_voices,
                    self.num_tokens_per_block // num_voices
                )

                x = {
                    'x_left':           x_left,
                    'x_right':          x_right,
                    'negative_samples': negative_sample
                }

                yield x

        dataloaders = [
            _aggregate_dataloader(dataloader)
            for dataloader
            in dataloaders
        ]

        return dataloaders

    def _dataloader_random_bad(self, batch_size, num_negative_samples):
        """
        Dataloader for negative_sampling_method == 'random_bad'
        uses the same dataset for positive and negative samples
        :param batch_size:
        :return:
        """

        # dataset should be initialized by self._dataset
        assert self.dataset is not None
        num_tokens_left = self.num_tokens_per_block * self.num_blocks_left
        num_tokens_per_beat = subdivision * num_voices

        # num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)
        positive_dataloaders = self.dataset.data_loaders(
            batch_size=batch_size
        )

        negative_dataloaders = self.dataset.data_loaders(
            batch_size=batch_size * num_negative_samples * self.num_blocks_right
        )

        # Generate dataloaders
        def _aggregate_dataloader(dataloader_positive,
                                  dataloader_negative):
            for p, n in zip(dataloader_positive, dataloader_negative):
                # remove metadata
                n = n[0]
                p = p[0]

                negative_sample = n[:, :, : self.num_tokens_per_block // num_voices]
                negative_sample = negative_sample.view(
                    batch_size,
                    num_negative_samples,
                    self.num_blocks_right,
                    num_voices,
                    self.num_tokens_per_block // num_voices
                )
                x_left = p[:, :, :num_tokens_left // num_voices]
                x_right = p[:, :, num_tokens_left // num_voices:]

                x = {
                    'x_left':           x_left,
                    'x_right':          x_right,
                    'negative_samples': negative_sample
                }

                yield x

        dataloaders = [
            _aggregate_dataloader(dataloader_positive, dataloader_negative)
            for dataloader_positive, dataloader_negative
            in zip(positive_dataloaders, negative_dataloaders)
        ]

        return dataloaders

    def _dataloader_random(self, batch_size, num_negative_samples):
        """
        Dataloader for negative_sampling_method == 'random'
        :param batch_size:
        :return:
        """

        # dataset should be initialized by self._dataset
        assert self.dataset is not None
        num_tokens_left = self.num_tokens_per_block * self.num_blocks_left
        num_tokens_per_beat = subdivision * num_voices

        # num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

        dataset_positive, dataset_negative = self.dataset
        positive_dataloaders = dataset_positive.data_loaders(
            batch_size=batch_size,
            cache_dir=self.cache_dir
        )

        # Negative dataset
        negative_dataloaders = dataset_negative.data_loaders(
            batch_size=batch_size * num_negative_samples * self.num_blocks_right,
            cache_dir=self.cache_dir
        )

        # Generate dataloaders
        def _aggregate_dataloader(dataloader_positive,
                                  dataloader_negative):
            for p, n in zip(dataloader_positive, dataloader_negative):
                # remove metadata
                negative_sample = n[0]
                p = p[0]
                assert negative_sample.size(2) == self.num_tokens_per_block // num_voices
                negative_sample = negative_sample.view(
                    batch_size,
                    num_negative_samples,
                    self.num_blocks_right,
                    num_voices,
                    self.num_tokens_per_block // num_voices
                )
                x_left = p[:, :, :num_tokens_left // num_voices]
                x_right = p[:, :, num_tokens_left // num_voices:]

                x = {
                    'x_left':           x_left,
                    'x_right':          x_right,
                    'negative_samples': negative_sample
                }

                yield x

        dataloaders = [
            _aggregate_dataloader(dataloader_positive, dataloader_negative)
            for dataloader_positive, dataloader_negative
            in zip(positive_dataloaders, negative_dataloaders)
        ]

        return dataloaders


class BachCPCSmallDataloaderGenerator(BachCPCDataloaderGenerator):
    def __init__(self,
                 num_tokens_per_block,
                 num_blocks_left,
                 num_blocks_right,
                 negative_sampling_method,
                 *args, **kwargs):
        """

        :param num_tokens_per_block:
        :param num_blocks_left:
        :param num_blocks_right:
        :param num_negative_samples:
        :param negative_sampling_method:
        :param args:
        :param kwargs:
        """
        assert num_tokens_per_block % (subdivision * num_voices) == 0
        super(BachCPCSmallDataloaderGenerator, self).__init__(
            num_tokens_per_block,
            num_blocks_left,
            num_blocks_right,
            negative_sampling_method)

    def _dataset(self):
        if self.negative_sampling_method == 'random':
            dataset_manager = DatasetManager()
            self.cache_dir = dataset_manager.cache_dir
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            chorale_dataset_positive_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_positive,
                'subdivision':    subdivision,
            }

            dataset_positive: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats_test',
                **chorale_dataset_positive_kwargs
            )
            num_tokens_per_beat = subdivision * num_voices
            num_beats_negative = self.num_tokens_per_block // num_tokens_per_beat
            chorale_dataset_negative_kwargs = {
                'voice_ids':      [0, 1, 2, 3],
                'metadatas':      metadatas,
                'sequences_size': num_beats_negative,
                'subdivision':    subdivision,
            }

            dataset_negative: ChoraleBeatsDataset = dataset_manager.get_dataset(
                name='bach_chorales_beats_test',
                **chorale_dataset_negative_kwargs
            )
            return dataset_positive, dataset_negative
        else:
            raise NotImplementedError
