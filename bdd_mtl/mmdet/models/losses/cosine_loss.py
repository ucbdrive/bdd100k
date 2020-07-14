import torch
import torch.nn as nn

from .cross_entropy_loss import _expand_binary_labels
from .utils import weight_reduce_loss
from ..registry import LOSSES


def cosine_embedding_loss(pred,
                          label,
                          weight=None,
                          margin=0.0,
                          balance_pos_neg=False,
                          reduction='mean',
                          avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(
            label, weight, pred.size(-1), balance_pos_neg=balance_pos_neg)
    loss = torch.where(label == 1, 1 - pred, torch.clamp(pred - margin, min=0))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module
class CosineEmbeddingLoss(nn.Module):

    def __init__(self,
                 margin=0.0,
                 balance_pos_neg=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.balance_pos_neg = balance_pos_neg
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        loss_cosine = self.loss_weight * cosine_embedding_loss(
            pred,
            target,
            weight=weight,
            margin=self.margin,
            balance_pos_neg=self.balance_pos_neg,
            reduction=self.reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cosine
