import torch


def mixing(data, layer_ind, mixup_lambdas, mixup_layers):
    # Todo instead of doubling the mixup, use opposite mixup
    batch_size, num_token, num_feature = data.size()

    mixup_lambdas_reshape = mixup_lambdas.unsqueeze(1).unsqueeze(2).repeat(1, num_token, num_feature)
    if layer_ind is not None:
        mixup_layers_reshape = mixup_layers.repeat(2).unsqueeze(1).unsqueeze(2).repeat(1, num_token, num_feature)

    mixed = mixup_lambdas_reshape * data[:batch_size // 2] + \
        (1 - mixup_lambdas_reshape) * data[batch_size // 2:]
    mixed = mixed.repeat(2, 1, 1)

    if layer_ind is None:
        # For mixing targets
        ret = mixed
    else:
        ret = torch.where(mixup_layers_reshape == layer_ind, mixed, data)

    return ret
