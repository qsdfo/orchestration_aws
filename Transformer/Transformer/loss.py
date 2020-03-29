import torch
import torch.nn.functional as F
from torch import nn


def mean_crossentropy(preds, targets, mask, shift, label_smoothing, loss_on_last_frame):
    """
    :param mask:
    :param ratio:
    :param targets: (batch, voice, chorale_length)
    :param preds: (num_instru, batch, nu_frames, num_notes) one for each voice
    since num_notes are different
    :return:
    """
    if mask is None:
        reduction = 'mean'
    else:
        reduction = 'none'

    cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
    loss = 0

    batch_size, length, num_voices = targets.size()

    targets_permute = targets.permute(2, 0, 1)
    if mask is not None:
        mask_permute = mask.permute(2, 0, 1)

    for voice_index, this_pred in enumerate(preds):
        this_target = targets_permute[voice_index]
        if mask is not None:
            this_mask = mask_permute[voice_index].float()

        if shift and (voice_index == 0):
            # If shifted sequences, don't use first prediction
            this_target = this_target[:, 1:]
            this_pred = this_pred[:, 1:, :]
            if mask is not None:
                this_mask = this_mask[:, 1:]
            flat_dim = batch_size * (length - 1)
        else:
            flat_dim = batch_size * length

        if loss_on_last_frame:
            this_target = this_target[:, -1]
            this_pred = this_pred[:, -1]
            flat_dim = batch_size

        this_targ_flat = this_target.contiguous().view(flat_dim)
        this_pred_flat = this_pred.contiguous().view(flat_dim, -1)
        if mask is not None:
            this_mask_flat = this_mask.view(flat_dim)

        if label_smoothing > 0:
            eps = label_smoothing
            n_class = this_pred_flat.size(1)
            one_hot = torch.zeros_like(this_pred_flat).scatter(1, this_targ_flat.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(this_pred_flat, dim=1)
            this_loss = -(one_hot * log_prb).sum(dim=1)
        else:
            this_loss = cross_entropy(this_pred_flat, this_targ_flat)

        if mask is not None:
            # Normalise by the number of elements actually used
            norm = this_mask_flat.sum() + 1e-20
            this_loss = torch.dot(this_loss, this_mask_flat) / norm
            # Or no Norm ? Harder examples (only 1 or two examples are not masked are favored then)
            # ce = torch.dot(ce, this_mask_flat)
        elif label_smoothing > 0:
            this_loss = this_loss.mean()

        loss += this_loss

    return loss


def mean_crossentropy_mixup(preds, targets, mask, shift, mixup_lambdas):
    """

    :param mask:
    :param ratio:
    :param targets: (batch, voice, chorale_length)
    :param preds: (num_instru, batch, nu_frames, num_notes) one for each voice
    since num_notes are different
    :return:
    """

    reduction = 'none'
    cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
    loss = 0

    batch_size, length, num_voices = targets.size()

    targets_0 = targets[:batch_size // 2].permute(2, 0, 1)
    targets_1 = targets[batch_size // 2:].permute(2, 0, 1)
    batch_size = batch_size // 2

    if mask is not None:
        mask_0 = mask[:batch_size].permute(2, 0, 1)
        mask_1 = mask[batch_size:].permute(2, 0, 1)
        raise Exception("Not checked, probably wrong")
    else:
        mask_0 = None
        mask_1 = None

    for voice_index, this_pred in enumerate(preds):

        def compute_ce(this_pred, this_targets, this_mask):
            this_target = this_targets[voice_index]
            if mask is not None:
                this_mask_voice = this_mask[voice_index].float()

            if shift and (voice_index == 0):
                # If shifted sequences, don't use first prediction
                this_target = this_target[:, 1:]
                this_pred = this_pred[:, 1:, :]
                if mask is not None:
                    this_mask_voice = this_mask_voice[:, 1:]
                flat_dim = batch_size * (length - 1)
                length_ce = length - 1
            else:
                flat_dim = batch_size * length
                length_ce = length

            this_targ_flat = this_target.contiguous().view(flat_dim)
            this_pred_flat = this_pred.contiguous().view(flat_dim, -1)
            if mask is not None:
                this_mask_flat = this_mask_voice.view(flat_dim)

            # Padding mask is one where input is masked, and these are the samples we want to use to backprop
            ce = cross_entropy(this_pred_flat, this_targ_flat)

            if mask is not None:
                # Normalise by the number of elements actually used
                norm = this_mask_flat.sum() + 1e-20
                ce = torch.dot(ce, this_mask_flat) / norm
            else:
                this_mask_flat = None

            ce = ce.view(batch_size, length_ce)

            return ce, this_mask_flat, length_ce

        ce_0, mask_flat_0, length_ce = compute_ce(this_pred, targets_0, mask_0)
        ce_1, mask_flat_1, length_ce = compute_ce(this_pred, targets_1, mask_1)

        if length_ce > 1:
            mixup_lambdas_reshape = mixup_lambdas.unsqueeze(1).repeat(1, length_ce)
        else:
            mixup_lambdas_reshape = mixup_lambdas

        loss += (mixup_lambdas_reshape * ce_0 + (1 - mixup_lambdas_reshape) * ce_1).mean()

    return loss
