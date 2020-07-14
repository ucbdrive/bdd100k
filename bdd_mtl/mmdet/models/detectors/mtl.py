import torch.nn as nn
from collections import defaultdict
import mmcv
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin, TrackTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (bbox2roi, bbox2result, results2track, track2results, segtrack2results, build_assigner,
                        build_sampler, tensor2imgs)
from spatial_correlation_sampler import SpatialCorrelationSampler as corr
from .. import builder
import torch
import numpy as np


@DETECTORS.register_module
class MTL(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin, TrackTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 cls_head=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 prop_track_roi_extractor=None,
                 prop_track_head=None,
                 asso_track_roi_extractor=None,
                 asso_track_head=None,
                 freeze_exclude_tracker=False,
                 corr_params=None,
                 lane_dir_head=None,
                 lane_sty_head=None,
                 lane_typ_head=None,
                 drivable_head=None,
                 sem_seg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 cfg=None,
                 pretrained=None):
        super(MTL, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.num_tasks = 0
        # specific heads
        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            self.num_tasks += 1
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
            self.num_tasks += 1
        if prop_track_head is not None:
            self.prop_track_head = builder.build_head(prop_track_head)
            if prop_track_roi_extractor is not None:
                self.prop_track_roi_extractor = builder.build_roi_extractor(
                    prop_track_roi_extractor)
        if asso_track_head is not None:
            self.asso_track_head = builder.build_head(asso_track_head)
            if asso_track_roi_extractor is not None:
                self.asso_track_roi_extractor = builder.build_roi_extractor(
                    asso_track_roi_extractor)
        if lane_dir_head is not None:
            self.lane_dir_head = builder.build_head(lane_dir_head)
            self.lane_sty_head = builder.build_head(lane_sty_head)
            self.lane_typ_head = builder.build_head(lane_typ_head)
        if drivable_head is not None:
            self.drivable_head = builder.build_head(drivable_head)
        if lane_dir_head is not None or drivable_head is not None:
            self.num_tasks += 1
        if sem_seg_head is not None:
            self.sem_seg_head = builder.build_head(sem_seg_head)
            self.num_tasks += 1

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        if prop_track_head is not None:
            self.corr = corr(**corr_params)
            self.freeze_exclude_tracker = freeze_exclude_tracker
            if self.freeze_exclude_tracker:
                print('Freezing modules except track head...')
                freeze_list = [
                    self.backbone, self.neck, self.rpn_head, self.bbox_head
                ]
                for sub_module in freeze_list:
                    sub_module.eval()
                    for param in sub_module.parameters():
                        param.requires_grad = False
    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_prop_track(self):
        return hasattr(self,
                       'prop_track_head') and self.prop_track_head is not None

    @property
    def with_asso_track(self):
        return hasattr(self,
                       'asso_track_head') and self.asso_track_head is not None

    @property
    def with_lane(self):
        return hasattr(self, 'lane_dir_head') and self.lane_dir_head is not None

    @property
    def with_drivable(self):
        return hasattr(self, 'drivable_head') and self.drivable_head is not None

    @property
    def with_sem_seg(self):
        return hasattr(self, 'sem_seg_head') and self.sem_seg_head is not None

    def init_weights(self, pretrained=None):
        super(MTL, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_prop_track:
            self.prop_track_roi_extractor.init_weights()
            self.prop_track_head.init_weights()
        if self.with_asso_track:
            self.asso_track_roi_extractor.init_weights()
            self.asso_track_head.init_weights()
        if self.with_lane:
            self.lane_dir_head.init_weights()
            self.lane_sty_head.init_weights()
            self.lane_typ_head.init_weights()
        if self.with_drivable:
            self.drivable_head.init_weights()
        if self.with_sem_seg:
            self.sem_seg_head.init_weights()

    def extract_feat(self, img):
        x0 = self.backbone(img)
        x1 = self.neck(x0[-4:]) if self.with_neck else None
        return x0, x1

    def extract_corr_feat(self, feat1, feat2):
        corr_x = [
            self.corr(_feat1, _feat2) for _feat1, _feat2 in zip(feat1, feat2)
        ]
        corr_x = [
            _corr_x.view(
                _corr_x.size(0), -1, _corr_x.size(3), _corr_x.size(4))
            for _corr_x in corr_x
        ]
        # c = 17 * 17 + 256 + 256 = 801
        track_x = [torch.cat(xs, dim=1) for xs in zip(feat1, feat2, corr_x)]
        return track_x

    def update(self, type='inplace', momentum=1.0, **kwargs):
        for k, v in kwargs.items():
            if type == 'inplace':
                setattr(self, k, v)
            elif type == 'contact':
                ori_v = getattr(self, k)
                setattr(self, k, torch.cat((ori_v, v), dim=0))
            elif type == 'update':
                ori_v = getattr(self, k)
                v = (1 - momentum) * ori_v + momentum * v
                setattr(self, k, v)
            elif type == 'max':
                ori_v = getattr(self, k)
                v = (v >= ori_v) * v + (v < ori_v) * ori_v
                setattr(self, k, v)
            else:
                raise TypeError('Not known type.')

    def forward_train(self,
                      img=None,
                      img_meta=None,
                      gt_cls=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_pids=None,
                      ref_img=None,
                      ref_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_lane=None,
                      gt_drivable=None,
                      gt_sem_seg=None,
                      return_vis=False):
        losses = dict()
        vis = dict()
        x0, x = self.extract_feat(img)
        # classification
        if self.with_cls_head and gt_cls is not None:
            cls_out = self.cls_head(x0[-1])
            losses.update(self.cls_head.loss(cls_out, gt_cls))

        # instance-based
        if gt_bboxes is not None or gt_masks is not None:
            # RPN forward and loss
            if self.with_rpn:
                rpn_outs = self.rpn_head(x)
                rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                              self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses.update(rpn_losses)

                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
                proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            else:
                proposal_list = proposals

            # assign gts and sample proposals
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    # tracking
                    if gt_pids is not None:
                        assign_result = bbox_assigner.assign(proposal_list[i],
                                                             gt_bboxes[i],
                                                             gt_bboxes_ignore[i],
                                                             gt_labels[i], gt_pids[i])
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            proposal_list[i],
                            gt_bboxes[i],
                            gt_labels[i],
                            gt_pids[i],
                            ref_bboxes[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                    else:
                        assign_result = bbox_assigner.assign(proposal_list[i],
                                                             gt_bboxes[i],
                                                             gt_bboxes_ignore[i],
                                                             gt_labels[i])
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            proposal_list[i],
                            gt_bboxes[i],
                            gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

        # detection-specific
        if gt_bboxes is not None and self.with_bbox:
            # bbox head forward and loss
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            if return_vis:
                for i in range(len(bbox_pred)):
                    vis['vis_det_info'] = bbox_pred
                    vis['vis_det_gt_info'] = gt_bboxes

        # instance-segmentation-specific
        if gt_masks is not None and self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        # track-specific
        if ref_img is not None:
            _, ref_x = self.extract_feat(ref_img)
            if self.with_prop_track:
                """Difference between this implementation and the
                    paper `Detect to Track and Track to Detect`:
                    1. Based on FPN Faster R-CNN instead of R-FCN, that means:
                        a. Correlations are implemented on specific levels.
                        b. Contacted features for TrackHead is from backbone/neck.
                    2. Training with positive proposals instead of only gts.
                """
                track_x = self.extract_corr_feat(x, ref_x)
                # TODO consider training use gt or (gt + props)[NOW]
                # TODO consider whether to include semantic consistence[NO]
                # TODO consider how to calculate the correlation features
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                prop_track_feats = self.prop_track_roi_extractor(
                    track_x[:self.prop_track_roi_extractor.num_inputs], pos_rois)
                prop_cls, prop_reg = self.prop_track_head(prop_track_feats)
                prop_targets = self.prop_track_head.get_target(
                    sampling_results, self.train_cfg.rcnn)
                loss_prop_track = self.prop_track_head.loss(
                    prop_cls, prop_reg, *prop_targets)
                losses.update(loss_prop_track)

            if self.with_asso_track:
                """Associate tracking, based on appearance features.
                """
                ref_rois = bbox2roi(ref_bboxes)
                num_bbox_x = [res.bboxes.size(0) for res in sampling_results]
                num_bbox_ref_x = [res.size(0) for res in ref_bboxes]
                bbox_feats = self.asso_track_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois)
                ref_bbox_feats = self.asso_track_roi_extractor(
                    ref_x[:self.asso_track_roi_extractor.num_inputs], ref_rois)
                asso_probs = self.asso_track_head(bbox_feats, ref_bbox_feats,
                                                  num_bbox_x, num_bbox_ref_x)
                asso_targets = self.asso_track_head.get_target(
                    sampling_results, self.train_cfg.track)
                loss_asso_track = self.asso_track_head.loss(
                    asso_probs, *asso_targets)
                losses.update(loss_asso_track)

        # lane
        if self.with_lane and gt_lane is not None:
            lane_dir_out = self.lane_dir_head(x[0])
            lane_sty_out = self.lane_sty_head(x[0])
            lane_typ_out = self.lane_typ_head(x[0])
            gt_lane = torch.stack(gt_lane)

            losses.update(self.lane_dir_head.loss(lane_dir_out, [gt_lane[:, :, :, 0].long()]))
            losses.update(self.lane_sty_head.loss(lane_sty_out, [gt_lane[:, :, :, 1].long()]))
            losses.update(self.lane_typ_head.loss(lane_typ_out, [gt_lane[:, :, :, 2].long()]))
            if return_vis:
                vis['vis_lane_dir'] = torch.argmax(lane_dir_out[0][0], dim=0).clone().cpu().numpy()
                vis['vis_lane_sty'] = torch.argmax(lane_sty_out[0][0], dim=0).clone().cpu().numpy()
                vis['vis_lane_typ'] = torch.argmax(lane_typ_out[0][0], dim=0).clone().cpu().numpy()

        # drivable
        if self.with_drivable and gt_drivable is not None:
            drivable_out = self.drivable_head(x[0])
            losses.update(self.drivable_head.loss(drivable_out, torch.stack(gt_drivable).unsqueeze(0).long()))
            if return_vis:
                for i in range(len(drivable_out)):
                    vis['vis_drivable_{}'.format(i)] = torch.argmax(drivable_out[i][0], dim=0).clone().cpu().numpy()

        # semantic seg
        if self.with_sem_seg and gt_sem_seg is not None:
            sem_seg_out = self.sem_seg_head(x[0])
            losses.update(self.sem_seg_head.loss(sem_seg_out, torch.stack(gt_sem_seg).unsqueeze(0).long()))
            if return_vis:
                for i in range(len(sem_seg_out)):
                    vis['vis_sem_seg_{}'.format(i)] = torch.argmax(sem_seg_out[i][0], dim=0).clone().cpu().numpy()
                losses, vis
        return losses

    def simple_test(self, img, img_meta, rescale=False, proposals=None):
        x0, x = self.extract_feat(img)
        out = dict()
        task = img_meta[0]['task']
        if self.with_cls_head and task == 'cls':
            cls_out = self.cls_head(x0[-1])
            out['cls_results'] = [np.argmax(c.data.cpu().numpy()) + 1 for c in cls_out]
        if self.with_lane and (task == 'street' or task == 'lane'):
            lane_dir_out = self.lane_dir_head(x[0])
            lane_sty_out = self.lane_sty_head(x[0])
            lane_typ_out = self.lane_typ_head(x[0])
            out['lane_results'] = np.array([lane_dir_out[0], lane_sty_out[0], lane_typ_out[0]])
        # drivable
        if self.with_drivable and (task == 'street' or task == 'drivable'):
            drivable_out = self.drivable_head(x[0])
            out['drivable_results'] = np.array([torch.argmax(d, dim=1).cpu().numpy() for d in drivable_out])
        # semantic seg
        if self.with_sem_seg and task == 'sem_seg':
            sem_seg_out = self.sem_seg_head(x[0])
            out['sem_seg_results'] = np.array([torch.argmax(d, dim=1).cpu().numpy() for d in sem_seg_out])
        # instance-based
        if task in ['det', 'ins_seg', 'box_track', 'seg_track']:
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
            out['bbox_results'] = bbox_results
            if task in ['ins_seg', 'seg_track']:
                out['segm_results'] = self.simple_test_mask(
                    x, img_meta, det_bboxes, det_labels, rescale=rescale)
            if task == 'box_track' or task == 'seg_track':
                track_bboxes = None
                if det_bboxes.nelement() == 0:
                    out['segm_track_results'] = defaultdict(list)
                    out['new_bbox_results'] = bbox_results,
                    out['track_results'] = defaultdict(list)
                    return out
                first_frame = img_meta[0]['first_frame']

                if first_frame or self.labels.shape[0] == 0:
                    new_bbox_results = bbox_results
                    track_results = self.setup_tracklets(
                        x, det_bboxes, det_labels, img_meta, rescale=rescale)
                else:
                    if self.with_prop_track:
                        prop_bboxes = self.simple_test_prop_track(
                            x, img_meta, rescale=rescale)
                    if self.with_asso_track:
                        asso_probs = self.simple_test_asso_track(
                            x, det_bboxes, det_labels, img_meta, rescale=rescale)
                    ids, new_det_bboxes, new_det_labels = self.assign_ids(
                        prop_bboxes,
                        asso_probs,
                        det_bboxes,
                        det_labels,
                    )
                    track_bboxes, track_labels, ids, det_bboxes, det_labels = self.update_tracklets(
                        x, ids, new_det_bboxes, new_det_labels)
                    track_results = track2results(track_bboxes, track_labels, ids)
                    new_bbox_results = bbox2result(det_bboxes, det_labels,
                                                   self.bbox_head.num_classes)
                if self.test_cfg.track.plot_track_results:
                    track_bboxes, track_ids = results2track(track_results)
                    self.plot_track_results(img, img_meta, track_bboxes, track_ids)
                out['new_bbox_results'] = new_bbox_results
                out['track_results'] = track_results
                if task == 'seg_track':
                    out['segm_track_results'] = defaultdict(list)
                    if first_frame:
                        track_bboxes = torch.Tensor([d['bbox'] for d in track_results.values()]).to(x[0].device)
                        track_labels = torch.LongTensor([d['label'] for d in track_results.values()]).to(x[0].device)
                        ids = torch.LongTensor(list(track_results.keys()))
                        # recover mask order by label
                        cls_ptrs = defaultdict(int)
                        track_masks = np.array([])
                        for l in track_labels.cpu().numpy():
                            track_masks = np.append(track_masks, [out['segm_results'][l][cls_ptrs[l]]])
                            cls_ptrs[l] += 1
                        out['segm_track_results'] = segtrack2results(track_bboxes, track_masks, track_labels, ids)
                    elif self.labels.shape[0] > 0 and track_bboxes is not None:
                        track_masks_out = self.simple_test_mask(
                            x, img_meta, track_bboxes, track_labels, rescale=rescale)
                        # recover mask order by label
                        cls_ptrs = defaultdict(int)
                        track_masks = []
                        for l in track_labels.cpu().numpy():
                            track_masks.append(track_masks_out[l][cls_ptrs[l]])
                            cls_ptrs[l] += 1
                        out['segm_track_results'] = segtrack2results(track_bboxes, track_masks, track_labels, ids)

        return out
