from pathlib import Path

subdivision = 4
num_voices = 4
num_beats = 2
config = {
    # ======= Model ========
    'embedding_size':          32,
    'encoding_size_zt':        64,
    'encoding_size_ct':        64,
    'corrupt_labels':          False,

    # ======= Encoder =======
    'bidirectional_enc':       False,
    'bidirectional_ar':        False,
    'rnn_hidden_size':         1024,
    'num_layers_enc':          2,
    'num_layers_ar':           2,
    'dropout':                 0,
    # ======== Vector quantizer =======
    'vector_quantizer_type':   'product',
    'vector_quantizer_kwargs': dict(
        num_codebooks=1,
        codebook_size=64,
        commitment_cost=0.25,
        use_batch_norm=False,
        squared_l2_norm=True
        # add corrupt indices
    ),

    # ======== Dataloader ======
    # Dataset
    'dataset_type':            'bach',
    'dataloader_kwargs':       dict(num_tokens_per_block=num_beats * subdivision * num_voices,
                                    num_blocks_left=5,
                                    num_blocks_right=5,
                                    negative_sampling_method='same_sequence'
                                    ),
    'subdivision':             subdivision,  # Number of frame per quarter note

    # ======== Training ========
    'lr':                      1e-3,
    'beta':                    0.01,  # multiplicative term in front of quantization loss
    'batch_size':              32,
    'num_negative_samples':    15,
    'num_batches':             64,
    'num_epochs':              2000,

    # ======== model ID ========
    'timestamp':               None,
    'savename':                Path(__file__).stem,
}