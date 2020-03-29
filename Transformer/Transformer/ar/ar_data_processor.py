import torch
from torch import nn

from Transformer.ar.ar_dataset import ARDataset
from Transformer.data_processor import DataProcessor
from Transformer.relative_attentions import RelativeSelfAttentionModule


class ARDataProcessor(DataProcessor):
    def __init__(self, dataset: ARDataset):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """
        # local_position_dim = d_ticks
        super(ARDataProcessor, self).__init__(dataset=dataset,
                                              embedding_dim=1)

        self.dataset = dataset
        self.flip_masks = False
        self.use_masks = True
        # labels and embeddings for local position
        self.positions = nn.Parameter(
            torch.Tensor([i for i in range(self.dataset.length)]).long(),
            requires_grad=False)
        self.position_embedding = nn.Embedding(self.dataset.length, 1)

        self.linear_output_notes = nn.ModuleList(
            [
                nn.Linear(8, 1)
                for _ in range(self.dataset.length)
            ]
        )
        return

    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        ret = torch.stack(tensors)
        return ret

    def get_len_max_seq(self):
        return self.dataset.length

    @staticmethod
    def prepare_target_for_loss(target):
        return target

    def embed(self, x):
        """
        :param x: (batch_size, time, num_voices)
        :return: seq: (batch_size, time * num_voices, embedding_size)
        """
        ret = x.repeat(1, 1, 7)
        eps = torch.distributions.Normal(0, 0.01)
        noise = cuda_variable(eps.sample(sample_shape=ret.shape))
        ret = ret + noise
        return ret
    #
    # def flatten(self, x):
    #     """
    #     :param x:(batch, num_frames, num_instruments, ...)
    #     :return: (batch, num_frames * num_instruments, ...) with num_instruments varying faster
    #     """
    #     batch_size = x.size()[0]
    #     # x = torch.reshape(x, (batch_size, self.num_frames * self.num_ticks_per_frame, -1))
    #     return ret
    #
    # def wrap(self, flatten_x):
    #     """
    #     Inverse of flatten operation
    #
    #     :param flatten_x: (batch_size, length, ...)
    #     :return:
    #     """
    #     batch_size = flatten_x.size(0)
    #     # x = torch.reshape(flatten_x, (batch_size, self.num_frames, self.num_ticks_per_frame, -1))
    #     return ret

    def mask_encoder(self, x):
        return x, None

    def mask_decoder(self, x, p):
        return x, None

    def mask_nade(self, x, epoch_id=None):
        return self.mask_UNIFORM_ORDER(x, epoch_id)

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return: preds: (num_instru, batch, num_frames, num_pitches)
        """
        # split voices
        pred_seq = pred_seq.permute(1, 0, 2)
        preds = [
            pre_softmax(pred)
            for pred, pre_softmax in zip(pred_seq, self.linear_output_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param batch_size:
        :param sequence_length:
        :return:
        """
        positions = self.positions.unsqueeze(0).repeat(batch_size, 1)[:, :sequence_length]
        embedded_positions = self.position_embedding(positions)
        return embedded_positions

    def get_relative_attention_module(self, embedding_dim,
                                      n_head,
                                      len_max_seq_cond,
                                      len_max_seq,
                                      use_masks,
                                      shift,
                                      enc_dec_attention):
        return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                           n_head=n_head,
                                           seq_len=len_max_seq,
                                           use_masks=use_masks)
