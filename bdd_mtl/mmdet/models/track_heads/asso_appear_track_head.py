import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule


def asso_bbox_target(pos_gt_ids_list, bboxes_list, target_cfg, concat=False):
    target_cfg = [target_cfg for i in range(len(bboxes_list))]
    ids, id_weights = multi_apply(asso_bbox_target_single, pos_gt_ids_list,
                                  bboxes_list, target_cfg)

    if concat:
        ids = torch.cat(ids, 0)
        id_weights = torch.cat(id_weights, 0)
    return ids, id_weights


def asso_bbox_target_single(pos_gt_ids, bboxes, target_cfg):
    use_neg = target_cfg['asso_use_neg']
    num_samples = bboxes.size(0)
    num_pos = pos_gt_ids.size(0)
    ids = pos_gt_ids.new_zeros(num_samples, dtype=torch.long)
    id_weights = pos_gt_ids.new_zeros(num_samples, dtype=torch.float)
    if num_pos > 0:
        ids[:num_pos] = pos_gt_ids + 1
        id_weights[:num_pos] = 1.0
    if use_neg:
        id_weights[num_pos:] = 1.0

    return ids, id_weights


@HEADS.register_module
class AssoAppearTrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference
       objects. Use dynamic option to deal with different number of objects
       in different images. A non-match entry is added to the reference
       objects with all-zero features. Object matched with the non-match entry
       is considered as a new object.
    """

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 residual_interval=-1,
                 last_norm=False,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_similarity=False,
                 loss_asso=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):
        super(AssoAppearTrackHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_asso.get('use_sigmoid', False)
        self.add_dummy = True if not loss_asso.get('use_sigmoid',
                                                   True) else False
        self.norm_similarity = norm_similarity
        self.residual_interval = residual_interval
        self.last_norm = last_norm
        if self.last_norm:
            self.norm = nn.GroupNorm(32, 1024)
        # For associtate loss, we consider:
        #   1. CrossEntropyLoss
        #   2. BinaryCrossEntropyLoss
        #   3. CosineEmbeddingLoss
        #   4. FocalLoss
        self.loss_asso = build_loss(loss_asso)
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_target(self, sampling_results, cfg):
        pos_gt_inds = [res.pos_gt_pids for res in sampling_results]
        bboxes = [res.bboxes for res in sampling_results]
        asso_targets = asso_bbox_target(pos_gt_inds, bboxes, cfg)
        return asso_targets

    def get_embeds(self, x):
        if x.nelement() == 0:
            return torch.zeros((0, self.fc_out_channels)).to(x.device)
        if self.num_convs > 0:
            _x = x
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if self.residual_interval > 0 and (
                        i + 1) % self.residual_interval == 0:
                    x = x + _x
                    _x = x
        if self.num_fcs > 0:
            x = x.view(x.size(0), -1)
            for i, fc in enumerate(self.fcs):
                x = fc(x)
                if i < self.num_fcs - 1:
                    x = self.relu(x)
                else:
                    if self.last_norm:
                        x = self.norm(x)
        return x

    def get_similarity(self, cur_embs, ref_embs):
        if self.norm_similarity:
            cur_embs = F.normalize(cur_embs, p=2, dim=1)
            ref_embs = F.normalize(ref_embs, p=2, dim=1)
        asso_probs = torch.mm(cur_embs, ref_embs.t())
        if self.add_dummy:
            m = asso_probs.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
            asso_probs = torch.cat([dummy, asso_probs], dim=1)
        return asso_probs

    def forward(self, x, ref_x, x_n, ref_x_n):
        assert len(x_n) == len(ref_x_n)
        # Siamese Features
        x = self.get_embeds(x)
        ref_x = self.get_embeds(ref_x)
        # Correlation use multiple
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        asso_probs = []
        for _x, _ref_x in zip(x_split, ref_x_split):
            asso_prob = self.get_similarity(_x, _ref_x)
            asso_probs.append(asso_prob)
        return asso_probs

    def loss(self, asso_probs, ids, id_weights):
        losses = dict()
        batch_size = len(ids)
        loss_asso = 0.
        match_acc = 0.
        n_total = 0
        # calculate per image loss
        for prob, cur_ids, cur_weights in zip(asso_probs, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0:
                continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            # TODO: check the avg_factor
            avg_factor = max(torch.sum(cur_weights > 0).float().item(), 1.)
            loss_asso += self.loss_asso(
                prob, cur_ids, cur_weights, avg_factor=avg_factor)
            if not self.add_dummy:
                dummy = torch.full((prob.size(0), 1), 0).to(prob.device)
                prob = torch.cat((dummy, prob), dim=1)
            match_acc += accuracy(
                torch.index_select(prob, 0, valid_idx),
                torch.index_select(cur_ids, 0, valid_idx)) * n_valid
        # average
        losses['loss_asso'] = loss_asso / batch_size
        if n_total > 0:
            losses['asso_acc'] = match_acc / n_total

        return losses
