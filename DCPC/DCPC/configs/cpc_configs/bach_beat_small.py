from pathlib import Path

subdivision = 4
num_voices = 4
num_beats = 1
config = {
    # ======= Model ========
    'embedding_size':          32,
    'encoding_size_zt':        64,
    'encoding_size_ct':        64,

    # ======= Encoder =======
    'bidirectional_enc':       False,
    'bidirectional_ar':        False,
    'rnn_hidden_size':         1024,
    'num_layers_enc':          2,
    'num_layers_ar':           2,
    'dropout':                 0.2,

    # ======== Vector quantizer =======
    # 'vector_quantizer_type':   'product',
    # 'vector_quantizer_kwargs': dict(
    #     num_codebooks=2,
    #     codebook_size=8,
    #     commitment_cost=0.25,
    #     use_batch_norm=True,
    #     squared_l2_norm=False
    #     # add corrupt indices
    # ),

    'vector_quantizer_type':   'none',
    'vector_quantizer_kwargs': dict(),

    # ======== Dataloader ======
    # Dataset
    # 'dataset_type':            'bach',
    # 'dataloader_kwargs':       dict(num_tokens_per_block=num_beats * subdivision * num_voices,
    #                                 num_blocks_left=5,
    #                                 num_blocks_right=4,
    #                                 negative_sampling_method='random'
    #                                 ),
    'dataset_type':            'bach_small',
    'dataloader_kwargs':       dict(num_tokens_per_block=num_beats * subdivision * num_voices,
                                    num_blocks_left=10,
                                    num_blocks_right=10,
                                    negative_sampling_method='random'
                                    ),
    'subdivision':             subdivision,  # Number of frame per quarter note

    # ======== Training ========
    'lr':                      1e-4,
    'batch_size':              8,
    'num_negative_samples':    7,
    'num_batches':             64,
    'num_epochs':              2000,

    # ======== model ID ========
    'timestamp':               None,
    'savename':                Path(__file__).stem,
}
