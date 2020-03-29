from pathlib import Path

subdivision = 4
num_voices = 4
num_beats = 1
config = {
    # ======= Model ========
    'embedding_size':          32,
    'encoding_size_zt':        128,
    'encoding_size_ct':        128,
    'corrupt_labels':          None,

    # ======= Encoder =======
    'bidirectional_enc':       False,
    'bidirectional_ar':        False,
    'rnn_hidden_size':         1024,
    'num_layers_enc':          2,
    'num_layers_ar':           2,
    'dropout':                 0.2,

    # ======== Vector quantizer =======
    'vector_quantizer_type':   'none',
    'vector_quantizer_kwargs': dict(),

    # ======== Dataloader ======
    # Dataset
    'dataset_type':            'bach',
    'dataloader_kwargs':       dict(num_tokens_per_block=num_beats * subdivision * num_voices,
                                    num_blocks_left=5,
                                    num_blocks_right=5,
                                    negative_sampling_method='random'
                                    ),
    'subdivision':             subdivision,  # Number of frame per quarter note

    # ======== Training ========
    'lr':                      1e-4,
    'beta': 0,  # multiplicative term in front of quantization loss
    'batch_size':              32,
    'num_negative_samples':    15,
    'num_batches':             64,
    'num_epochs':              2000,

    # ======== model ID ========
    'timestamp':               None,
    'savename':                Path(__file__).stem,
}
