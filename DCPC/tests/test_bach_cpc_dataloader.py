from DCPC.dataloaders.bach_cpc_dataloader import BachCPCDataloaderGenerator


def test_random_dataloader():
    num_tokens_per_block = 16
    num_blocks_left = 4
    num_blocks_right = 5
    negative_sampling_method = 'random'

    dataloader = BachCPCDataloaderGenerator(
        num_tokens_per_block=num_tokens_per_block,
        num_blocks_left=num_blocks_left,
        num_blocks_right=num_blocks_right,
        negative_sampling_method=negative_sampling_method
    )

    batch_size = 32
    num_negative_samples = 15
    (dataloader_train,
     dataloader_val,
     dataloader_test) = dataloader.dataloader(batch_size=batch_size,
                                              num_negative_samples=num_negative_samples
                                              )

    x = next(dataloader_train)
    print(x['x_left'].shape)
    print(x['x_right'].shape)
    print(x['negative_samples'].shape)


if __name__ == '__main__':
    test_random_dataloader()
