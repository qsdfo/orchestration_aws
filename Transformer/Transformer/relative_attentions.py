import torch

from torch import nn

from Transformer.helpers import cuda_variable


class RelativeSelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim, n_head, seq_len, use_masks):
        # todo
        #  TIED PARAMETERS
        #  embedding dim = d_k
        super(RelativeSelfAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.use_masks = use_masks

        if self.use_masks:
            self.e = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
        else:
            self.e1 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
            self.e2 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        assert sz_b_times_n_head % self.n_head == 0
        sz_b = sz_b_times_n_head // self.n_head

        if self.use_masks:
            # In this case,
            # rel_attn will be masked in RelativeScaledDotProductAttention
            # the size of e can be larger that the size of q
            # todo add shift argument?!
            e = self.e[:self.n_head * len_q, :]
            e = e.unsqueeze(0).repeat(sz_b, 1, 1)
            e = e.view(sz_b * self.n_head, len_q, d_q)

            rel_attn = torch.einsum('bld,bmd->blm',
                                    (q, e))
            batch_size, l, _ = rel_attn.size()
            # skewing trick
            if flipped_masks:
                # pad
                rel_attn = torch.cat(
                    [rel_attn,
                     cuda_variable((torch.ones(batch_size, l, 1, ) * - 100)),
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, :-1, :]
            else:
                # todo refine
                # pad
                rel_attn = torch.cat(
                    [cuda_variable((torch.ones(batch_size, l, 1, ) * - 100)),
                     rel_attn
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, 1:, :]

        # if no mask is used
        # the trick must be done twice
        else:
            e1 = self.e1.unsqueeze(0).repeat(sz_b, 1, 1)
            e1 = e1.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_1 = torch.einsum('bld,bmd->blm',
                                      (q, e1))
            e2 = self.e2.unsqueeze(0).repeat(sz_b, 1, 1)
            e2 = e2.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_2 = torch.einsum('bld,bmd->blm',
                                      (q, e2))

            batch_size, l, _ = rel_attn_1.size()
            # ====skewing trick
            # ----Down
            # pad
            rel_attn_1 = torch.cat(
                [rel_attn_1,
                 cuda_variable((torch.ones(batch_size, l, 1, ) * - 100)),
                 ], dim=2
            )
            rel_attn_1 = rel_attn_1.view(batch_size, l + 1, l)

            rel_attn_1 = rel_attn_1[:, :-1, :]

            # ----Up

            # pad
            rel_attn_2 = torch.cat(
                [cuda_variable((torch.ones(batch_size, l, 1, ) * - 100)),
                 rel_attn_2
                 ], dim=2
            )
            rel_attn_2 = rel_attn_2.view(batch_size,
                                         l + 1,
                                         l,
                                         )

            rel_attn_2 = rel_attn_2[:, 1:, :]

            masks_down = torch.triu(torch.ones_like(rel_attn_1[0]).byte(),
                                    diagonal=0).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).flip(
                1).flip(2).type(torch.bool)
            masks_up = torch.triu(torch.ones_like(rel_attn_2[0]).byte(),
                                  diagonal=1).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).type(torch.bool)

            rel_attn_1 = rel_attn_1.masked_fill(masks_down, 0)
            rel_attn_2 = rel_attn_2.masked_fill(masks_up, 0)
            rel_attn = rel_attn_1 + rel_attn_2
        return rel_attn


class NoPositionwiseAttention(nn.Module):
    #  todo
    #  Zero attention module here ? Perhaps try the block matrix stuff also here
    def __init__(self, seq_len_enc):
        super(NoPositionwiseAttention, self).__init__()
        self.seq_len_enc = seq_len_enc
        return

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        rel_attn = torch.zeros(sz_b_times_n_head, len_q, self.seq_len_enc)
        return rel_attn


class BlockSelfAttentionModule(nn.Module):
    """

    Block Self-attention.
    Blocks per time and per voice
    """

    def __init__(self, embedding_dim, n_head, dim_in, dim_out, num_frames_in, num_frames_out, use_masks,
                 use_voice_attention, shift):
        super(BlockSelfAttentionModule, self).__init__()

        self.n_head = n_head
        self.num_frames_in = num_frames_in
        self.num_frames_out = num_frames_out
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.length_in = num_frames_in * dim_in
        self.length_out = num_frames_out * dim_out
        self.embedding_dim = embedding_dim
        self.shift = shift
        self.use_masks = use_masks
        self.use_voice_attention = use_voice_attention

        # Parameters
        self.r_voice = nn.Parameter(torch.randn(dim_in, dim_out, embedding_dim, n_head))
        self.max_num_frames = max(num_frames_in, num_frames_out)
        self.e_past = nn.Parameter(torch.randn(self.max_num_frames, embedding_dim, n_head))
        if not use_masks:
            #  Also need negatives relative values in that case
            self.e_future = nn.Parameter(torch.randn(self.max_num_frames, embedding_dim, n_head))
        return

    def preprocess(self):
        #  Masks for grouping e0 and e1
        if not self.use_masks:
            one = torch.ones(self.max_num_frames, self.max_num_frames)
            mask_past = (torch.tril(one, diagonal=0)
                         .unsqueeze(2).unsqueeze(3)
                         .repeat(1, 1, self.embedding_dim, self.n_head))
            mask_future = (torch.triu(one, diagonal=1)
                           .unsqueeze(2).unsqueeze(3)
                           .repeat(1, 1, self.embedding_dim, self.n_head))
            # Note that with the way we compute S_rel, one value of e_future is "wasted"
        else:
            mask_past = None

        ############################################
        ############################################
        #  TEST VALUES
        # r_voice = torch.zeros_like(self.r_voice)
        # e_past = torch.zeros_like(self.e_past)
        # e_future = torch.zeros_like(self.e_future)
        # for i in range(self.dim_in):
        #     for j in range(self.dim_out):
        #         r_voice[i, j] = (i + 1) * 10 + (j + 1) * 100
        # for i in range(self.max_num_frames):
        #     e_past[i] = i + 1
        #     e_future[i] = -(i + 1)
        ############################################
        ############################################

        r_voice = self.r_voice
        e_past = self.e_past
        if not self.use_masks:
            e_future = self.e_future

        r_voice_repeat = r_voice.repeat(self.num_frames_in, self.num_frames_out, 1, 1)
        if self.shift:
            # Remove first column and last row
            # r_voice_repeat = r_voice_repeat[1:, :-1]
            r_voice_repeat = r_voice_repeat[:-1, :-1]

        def prepare_r_time(inp, mask, is_future=False):
            # triangularise
            a = torch.cat((inp, (torch.zeros(1, self.embedding_dim, self.n_head) - 111)), 0)
            b = a.unsqueeze(0).repeat(self.max_num_frames, 1, 1, 1)
            c = b.view((self.max_num_frames + 1, self.max_num_frames, self.embedding_dim, self.n_head))
            # d = c[:self.max_num_frames]
            if is_future:
                d = c[:self.max_num_frames]
            else:
                d = c[:self.max_num_frames].permute(1, 0, 2, 3)

            def expand(mat):
                length_out = self.dim_out * self.max_num_frames
                length_in = self.dim_in * self.max_num_frames
                # e = mat.repeat(1, self.dim_out, 1, 1)
                # f = e.view(length_out, self.max_num_frames, self.embedding_dim, self.n_head)
                # g = f.permute(1, 0, 2, 3)
                # h = g.repeat(1, self.dim_in, 1, 1)
                # ret = h.view(length_in, length_out, self.embedding_dim, self.n_head).permute(1, 0, 2, 3)
                e = mat.repeat(1, self.dim_in, 1, 1)
                f = e.view(length_in, self.max_num_frames, self.embedding_dim, self.n_head)
                g = f.permute(1, 0, 2, 3)
                h = g.repeat(1, self.dim_out, 1, 1)
                ret = h.view(length_out, length_in, self.embedding_dim, self.n_head).permute(1, 0, 2, 3)
                return ret

            r_time = expand(d)
            #  Truncate
            r_time_trunc = r_time[:self.length_in, :self.length_out, :, :]

            #  Matrices are expanded from permuted inputs... but it's okay, just believe it, it works :)
            if mask is not None:
                mask_expanded = expand(mask)
                mask_expanded_trunc = mask_expanded[:self.length_in, :self.length_out, :, :]
            else:
                mask_expanded_trunc = None

            return r_time_trunc, mask_expanded_trunc

        r_time_past, mask_expanded_past = prepare_r_time(e_past, mask_past)
        if not self.use_masks:
            r_time_future, mask_expanded_future = prepare_r_time(e_future, mask_future, is_future=True)
            r_time = mask_expanded_past * r_time_past + mask_expanded_future * r_time_future
        else:
            r_time = r_time_past

        if self.shift:
            r_time = r_time[:-1, :-1]

        return r_voice_repeat, r_time

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        # TODO use different q for time and instrument
        q_instru = q
        q_time = q
        sz_b = int(q.size()[0] / self.n_head)

        def prepare_r_matrices(r_matrix):
            r_matrix = r_matrix.permute(3, 0, 1, 2)
            # Batch is moving faster in q, so repeat like this is okay here
            r_matrix = r_matrix.repeat(sz_b, 1, 1, 1)
            return r_matrix

        r_voice_preprocessed, r_time_preprocessed = self.preprocess()
        r_voice = prepare_r_matrices(r_voice_preprocessed)
        r_time = prepare_r_matrices(r_time_preprocessed)

        rel_attn_instrument = torch.einsum('bld,blmd->blm', (q_instru, r_voice))
        rel_attn_time = torch.einsum('bld,blmd->blm', (q_time, r_time))

        if self.use_voice_attention:
            rel_attn = rel_attn_instrument + rel_attn_time
        else:
            #  For piano to orchestra self attention
            rel_attn = rel_attn_time

        return rel_attn

# OLD ONE
# DOES NOT USE THE TRICK FOR MAGENTA
# TODO adapt signature
# AND IT USED ONLY ONE SHARED R AMONGST THE HEADS
# class BachRelativeAttentionModule(nn.Module):
#     def __init__(self, embedding_dim, max_len_seq):
#         # todo embedding dim = d_k
#         super(BachRelativeAttentionModule, self).__init__()
#         self.embedding_dim = embedding_dim
#         # share R between heads
#         self.w_rs_tick = nn.Embedding(max_len_seq // 2 + 1, self.embedding_dim,
#                                       padding_idx=0)
#         self.w_rs_mod = nn.Embedding(17 + 1, self.embedding_dim,
#                                      padding_idx=0)
#
#         # TODO don't use triangular r?!
#         # self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + 1 if i - j >= 0 else 0 for i in
#         #                                    range(
#         #                                        get_len_max_seq)]
#         #                                   for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + max_len_seq // 4 + 1 for i in
#                                            range(max_len_seq)]
#                                           for j in range(max_len_seq)]).long().to('cuda')
#         # self.r_index_mod = torch.Tensor([[((i % 4) * 4 + (j % 4)) if i - j >= 0 else -1 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         # not the best idea:
#         undirected_order_dict = {(0, 0): 0,
#                                  (0, 1): 1,
#                                  (1, 0): 1,
#                                  (0, 2): 2,
#                                  (2, 0): 2,
#                                  (0, 3): 3,
#                                  (3, 0): 3,
#                                  (1, 1): 4,
#                                  (1, 2): 5,
#                                  (2, 1): 5,
#                                  (1, 3): 6,
#                                  (3, 1): 6,
#                                  (2, 2): 7,
#                                  (2, 3): 8,
#                                  (3, 2): 8,
#                                  (3, 3): 9,
#                                  }
#         # self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1 if i - j >= 0
#         #                                   else 0 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1
#                                           for i in
#                                           range(
#                                               max_len_seq)]
#                                          for j in range(max_len_seq)]).long().to('cuda')
#
#     def forward(self, sequence_length, n_head, batch_size):
#         # r is shared amongst the heads
#         r_mod = self.w_rs_mod(self.r_index_mod[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#         r_tick = self.w_rs_tick(self.r_index_tick[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#
#         r_mod = r_mod.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_mod = r_mod.view(-1, sequence_length, sequence_length,
#                            self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r_tick = r_tick.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_tick = r_tick.view(-1, sequence_length, sequence_length,
#                              self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r = r_mod + r_tick
#         return r
