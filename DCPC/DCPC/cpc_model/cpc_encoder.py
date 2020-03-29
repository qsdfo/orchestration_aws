import torch
from torch import nn
import numpy as np


class CPC_encoder(nn.Module):
    def __init__(self,
                 embedding_size,
                 #
                 num_layers_enc,
                 encoding_size_zt,  # TODO name this z_dim
                 bidirectional_enc,
                 #
                 rnn_hidden_size,
                 dropout,
                 ):
        super().__init__()

        self.bidirectional_enc = bidirectional_enc

        #  Model
        self.g_enc_fwd = torch.nn.GRU(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers_enc,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        if bidirectional_enc:
            self.g_enc_bwd = torch.nn.GRU(
                input_size=embedding_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers_enc,
                bias=True,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )

        linear_input_dim = rnn_hidden_size * 2 if bidirectional_enc else rnn_hidden_size
        self.post_rnn_linear_z = torch.nn.Linear(linear_input_dim, encoding_size_zt, bias=True)

    def forward(self, inputs):
        return self.compute_z(inputs)

    def compute_z(self, x):
        seq_len, input_size = x.size()[-2:]
        batch_dims = x.size()[:-2]
        product_batch_dims = np.prod(batch_dims)

        x = x.view(product_batch_dims, seq_len, input_size)
        z, _ = self.g_enc_fwd(x)
        # take last
        z = z[:, -1]
        # linear
        z = self.post_rnn_linear_z(z)

        z = z.view(*batch_dims, -1)
        return z
