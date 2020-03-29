import random

import numpy as np
import torch
from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from torch import nn

from Transformer.bach.bach_data_processor import BachBeatsDataProcessor


class BachBeatsCPCDataProcessor(BachBeatsDataProcessor):
    def __init__(self, dataset: ChoraleBeatsDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 encoder_flag,
                 monophonic_flag,
                 nade_flag,
                 cpc_model
                 ):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """

        embedding_dim = embedding_dim - cpc_model.encoding_size_zt

        super(BachBeatsCPCDataProcessor, self).__init__(
            dataset=dataset,
            embedding_dim=embedding_dim,
            reducer_input_dim=reducer_input_dim,
            local_position_embedding_dim=local_position_embedding_dim,
            encoder_flag=encoder_flag,
            monophonic_flag=monophonic_flag,
            nade_flag=nade_flag)

        # cpc model
        self.cpc_model = cpc_model

        self.num_tokens_per_block = self.cpc_model.dataloader_generator.num_tokens_per_block
        self.num_ticks_per_block = self.num_tokens_per_block // self.num_voices

        # Overwrite sequences length0
        max_len_sequence = self.dataset.sequences_size * self.dataset.subdivision * self.num_voices - self.num_tokens_per_block
        self.max_len_sequence_encoder = max_len_sequence
        if not nade_flag:
            self.max_len_sequence_decoder = max_len_sequence // 2
        else:
            self.max_len_sequence_decoder = max_len_sequence

    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        # load data, compute CPC
        x = tensors[0]
        if self.nade_flag or self.encoder_flag:
            #  Use only the melody ?
            # ret = x[:, 0:1, :]
            ret = x[:, :, self.num_ticks_per_block:]
        else:
            ret = x[:, :, self.num_ticks_per_block:]
            batch, voices, length = ret.size()
            ret = ret[:, :, :length // 2]

        # CPC embedding
        # Todo which one are we using here ? z, zq, re-embed encodings ?
        x_cpc = self.cpc_model.dataprocessor.preprocess({'x': x})
        x_cpc = self.cpc_model.dataprocessor.embed(x_cpc)
        z = self.cpc_model.encoder.compute_z(x_cpc['x'])
        (z_quantized,
         encodings,
         _) = self.cpc_model.vector_quantizer(
            z,
            corrupt_labels=False)

        #  Shift cpc codes
        # The cpc code of the previous code is associated to a given block of input
        cpc_code = z_quantized[:, :-1]
        batch_size, num_blocks, z_dim = cpc_code.shape

        if not (self.nade_flag or self.encoder_flag):
            cpc_code = cpc_code[:, :num_blocks // 2]
            num_blocks = num_blocks // 2

        # Duplicate along block dim
        cpc_code = cpc_code \
            .unsqueeze(2) \
            .repeat(1, 1, self.num_tokens_per_block, 1) \
            .view(batch_size, num_blocks * self.num_tokens_per_block, z_dim)

        return ret.long().cuda(non_blocking=True), cpc_code

    def get_len_max_seq(self):
        return self.max_len_sequence

    @staticmethod
    def prepare_target_for_loss(target):
        return target.permute(0, 2, 1)

    @staticmethod
    def prepare_mask_for_loss(mask):
        return mask.permute(0, 2, 1)

    def embed(self, x):
        """
        :param x: (batch_size, num_voices, chorale_length)
        :return: seq: (batch_size, chorale_length * num_voices, embedding_size)
        """
        separate_voices = x.split(split_size=1, dim=1)
        separate_voices = [
            embedding(voice[:, 0, :])[:, None, :, :]
            for voice, embedding
            in zip(separate_voices, self.note_embeddings)
        ]
        x = torch.cat(separate_voices, 1)
        x = self.flatten(x=x)
        return x

    def flatten(self, x):
        """

        :param x:(batch, num_voices, chorale_length, ...)
        :return: (batch, num_voices * chorale_length, ...) with num_voices varying faster
        """
        size = x.size()
        assert len(size) >= 3
        batch_size, num_voices, chorale_length = size[:3]
        remaining_dims = list(size[3:])
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_voices * chorale_length, *remaining_dims)
        return x

    def mask_encoder(self, x, p):
        #  Encoder part, we just want to remove information, randomly
        return self.mask_UNIFORM_ORDER(x)

    def mask_decoder(self, x, p):
        # This is more like input dropout, to enforce de decoder to use the encoder information
        return self.mask(x, p)

    # TODO Not implemented!!!
    def mask_nade(self, x, epoch_id):
        return self.mask_UNIFORM_ORDER(x)

    def mask_UNIFORM_ORDER(self, x):
        # Order is sampled first, then mask built as permutation
        # Masking done on all frames, no scheduling...
        batch_size, num_voices, length = x.size()

        # Order is sampled uniformly
        flat_dim = num_voices * length
        masks_non_shuffled = np.zeros((batch_size, flat_dim))
        lower_proba = int(flat_dim // 3)
        higher_proba = flat_dim + 1
        num_masked_events = np.random.randint(low=lower_proba, high=higher_proba, size=(batch_size))  #   0 is useless
        for batch_index in range(batch_size):
            masks_non_shuffled[batch_index, :num_masked_events[batch_index]] = 1

        mask_np = (np.random.permutation(masks_non_shuffled.T)).T
        mask_reshape = mask_np.reshape((batch_size, num_voices, length))
        mask = torch.tensor(mask_reshape).long().cuda()  #  1 means masked

        #  Mask symbol is the last token (self.num_notes_per_voice)
        mask_symbol_matrix = torch.Tensor(self.num_notes_per_voice).long().cuda()
        mask_symbol_matrix = mask_symbol_matrix.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, length)

        masked_chorale = x.clone() * (1 - mask) + mask_symbol_matrix.clone() * mask
        return masked_chorale, mask

    def mask(self, x, p):
        if p is None:
            p = random.random() / 2 + 0.5

        batch_size, num_voices, length = x.size()
        mask = (torch.rand_like(x.float()) < p).long()
        nc_indexes = torch.Tensor(self.num_notes_per_voice).long().cuda()
        nc_indexes = nc_indexes.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, length)

        masked_chorale = x.clone() * (1 - mask) + nc_indexes * mask
        return masked_chorale, mask

    @staticmethod
    def mean_crossentropy(pred, target, ratio=None):

        """

        :param pred: list (batch, chorale_length, num_notes) one for each voice
        since num_notes are different
        :param target:(batch, voice, chorale_length)
        :return:
        """
        # TODO use ratio argument?
        cross_entropy = nn.CrossEntropyLoss(size_average=True)
        sum = 0
        # put voice first for targets
        targets = target.permute(1, 0, 2)
        for voice_index, (voice_weight, voice_target) in enumerate(zip(pred, targets)):
            # put time first
            voice_weight = voice_weight.permute(1, 0, 2)
            voice_target = voice_target.permute(1, 0)
            for time_index, (w, t) in enumerate(zip(voice_weight, voice_target)):
                if time_index == 0 and voice_index == 0:  # exclude first dummy prediction
                    # if time_index < 12: # exclude first predictions
                    continue
                ce = cross_entropy(w, t)
                sum += ce

        return sum

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return:
        """
        batch_size, length, num_features = pred_seq.size()

        assert length % 4 == 0
        # split voices
        pred_seq = pred_seq.view(batch_size,
                                 length // 4, 4, num_features).permute(0, 2, 1, 3)
        # pred_seq (b, num_voices, chorale_length, num_features)
        preds = pred_seq.split(1, dim=1)
        preds = [
            pre_softmax(pred[:, 0, :, :])
            for pred, pre_softmax in zip(preds, self.linear_ouput_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param sequence_length:
        :return:
        """
        num_repeats = sequence_length // self.num_local_positions + 1
        positions = self.instrument_labels.unsqueeze(0).repeat(batch_size, num_repeats)[:,
                    :sequence_length]
        embedded_positions = self.instrument_embedding(positions)
        return embedded_positions