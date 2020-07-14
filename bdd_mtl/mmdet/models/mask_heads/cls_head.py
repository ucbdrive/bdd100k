import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target, force_fp32, auto_fp16


@HEADS.register_module
class ClsHead(nn.Module):

    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=[6, 6, 3],
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss')):
        super(ClsHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)

        self.convs = nn.ModuleList()
        for _ in range(3):
            convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = (
                    self.in_channels if i == 0 else self.conv_out_channels)
                padding = (self.conv_kernel_size - 1) // 2
                convs.append(
                    ConvModule(
                        in_channels,
                        self.conv_out_channels,
                        self.conv_kernel_size,
                        stride=2,
                        padding=padding,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
            self.convs.append(convs)


        self.fcs = nn.ModuleList([nn.Linear(conv_out_channels, n) for n in num_classes])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)

    @auto_fp16()
    def forward(self, x):
        x = x.unsqueeze(0).expand(3, -1, -1, -1, -1)
        for i in range(len(self.convs[0])):
            x = [convs[i](_x) for _x, convs in zip(x, self.convs)]
        # global avg pool
        x = [torch.mean(_x.view(_x.size(0), _x.size(1), -1), dim=2) for _x in x]
        preds = [fc(_x) for _x, fc in zip(x, self.fcs)]
        return preds

    @force_fp32(apply_to=('preds', ))
    def loss(self, preds, labels):
        loss = dict()
        loss_cls = 0
        for i in range(len(labels[0])):
            # if i != 2:
            #     continue
            labels_i = torch.stack([l[i] for l in labels])
            loss_cls += self.loss_cls(preds[i], labels_i - 1)
        loss['loss_cls'] = loss_cls
        return loss
