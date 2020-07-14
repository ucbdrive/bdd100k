import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES


def cross_entropy(pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100):
    # element-wise losses
    loss = F.cross_entropy(pred, label, class_weight, reduction='none', ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels,
                          label_weights,
                          label_channels,
                          balance_pos_neg=False):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        if not balance_pos_neg or label_channels == 1:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)
        else:
            bin_label_weights = label_weights.new_full(
                (label_weights.size(0), label_channels), 0)
            bin_label_weights[inds, :] = 1 / (label_channels - 1)
            bin_label_weights[inds, labels[inds] - 1] = 1.0

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         class_weight=None,
                         reduction='mean',
                         avg_factor=None,
                         balance_pos_neg=False,
                         ignore_index=-100):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(
            label, weight, pred.size(-1), balance_pos_neg=balance_pos_neg)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 balance_pos_neg=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.balance_pos_neg = balance_pos_neg

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self.ignore_index = ignore_index

    def forward(self,
                cls_score,
                label,
                weight=None,
                class_weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            kwargs.update(balance_pos_neg=self.balance_pos_neg)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index,
            **kwargs)
        return loss_cls
