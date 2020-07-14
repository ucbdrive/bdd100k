import os
from collections import defaultdict
import mmcv
import torch
import torch.nn as nn
from mmdet.core import (bbox2roi, bbox2result, results2track, build_assigner,
                        build_sampler, tensor2imgs)
from spatial_correlation_sampler import SpatialCorrelationSampler as corr
from .base import BaseDetector
from .cotest_mixins import (RPNTestMixin, BBoxTestMixin, TrackTestMixin)
from .. import builder
from ..registry import DETECTORS


# TODO implement mask and aug_test
@DETECTORS.register_module
class COTDN(BaseDetector, RPNTestMixin, BBoxTestMixin, TrackTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 prop_track_roi_extractor=None,
                 prop_track_head=None,
                 asso_track_roi_extractor=None,
                 asso_track_head=None,
                 affinity_head=None,
                 freeze_exclude_tracker=False,
                 corr_params=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 cfg=None,
                 pretrained=None):
        super(COTDN, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

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

        if affinity_head is not None:
            self.affinity_head = builder.build_head(affinity_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = cfg

        self.init_weights(pretrained=pretrained)

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

    def init_weights(self, pretrained=None):
        super(COTDN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_prop_track:
            self.prop_track_roi_extractor.init_weights()
            self.prop_track_head.init_weights()
        if self.with_asso_track:
            self.asso_track_roi_extractor.init_weights()
            self.asso_track_head.init_weights()
        if hasattr(self, 'affinity_head'):
            self.affinity_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_pids,
                      ref_img,
                      ref_bboxes,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)

        losses = dict()

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
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)


            affinity = self.affinity_head(x, ref_x, proposal_list,
                                          sampling_results)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

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

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        self.frame_id = img_meta[0]['frame_id']
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        # feature extraction for current frame
        x = self.extract_feat(img)
        # generate region proposals for current frame
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        # init tracklets if first frame or the tracklets are not initialized
        first_frame = img_meta[0]['first_frame']
        if first_frame or self.labels.shape[0] == 0:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x,
                img_meta,
                proposal_list,
                self.test_cfg.rcnn,
                rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
            if det_bboxes.nelement() == 0:
                return dict(
                    bbox_results=bbox_results,
                    new_bbox_results=bbox_results,
                    track_results=defaultdict(list))
            new_bbox_results = bbox_results
            track_results = self.setup_tracklets(
                x,
                det_bboxes,
                det_labels,
                img_meta,
                cfg=self.test_cfg.track,
                rescale=rescale)
        else:
            """Original Object Detection"""
            det_bboxes, det_labels = self.simple_test_bboxes(
                x,
                img_meta,
                proposal_list,
                self.test_cfg.rcnn,
                rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
            """For Synergistic Object Detection and Tracking"""
            if self.with_prop_track:
                # !!!prop_bboxes are based on scaled corrdinates.
                prop_bboxes = self.simple_test_prop_track(
                    x, img_meta, rescale=rescale)
            if self.with_asso_track:
                asso_probs = self.simple_test_asso_track(x, proposal_list)
            affinity = self.affinity_head(proposal_list[0], prop_bboxes,
                                          asso_probs, self.test_cfg.track,
                                          self.vanish_frames, self.prev_ids)
            new_det_bboxes, new_det_labels, ids = self.simple_syner_test(
                x, proposal_list, affinity, img_meta, self.test_cfg.rcnn,
                rescale)
            asso_probs = self.simple_test_asso_track_old(
                x, new_det_bboxes, img_meta, rescale)
            ids = self.assign_ids(x, prop_bboxes, asso_probs, new_det_bboxes,
                                  new_det_labels, img_meta, rescale)
            new_bbox_results = bbox2result(new_det_bboxes, new_det_labels,
                                           self.bbox_head.num_classes)
            track_results = self.update_tracklets(x, ids, new_det_bboxes,
                                                  new_det_labels, img_meta,
                                                  rescale)

        if self.test_cfg.track.plot_track_results:
            track_bboxes, track_ids = results2track(track_results)
            self.plot_track_results(img, img_meta, track_bboxes, track_ids)

        output = dict(
            bbox_results=bbox_results,
            new_bbox_results=new_bbox_results,
            track_results=track_results)
        return output

    def plot_track_results(self, img, img_meta, track_bboxes, track_ids):
        img = tensor2imgs(img, **self.cfg.img_norm_cfg)[0]
        out_folder = os.path.join(self.cfg.out_path,
                                  str(img_meta[0]['video_id']))
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(
            out_folder, '{}.png'.format(str(img_meta[0]['frame_id']).zfill(3)))
        img = mmcv.imshow_track_bboxes(
            img,
            track_bboxes,
            track_ids,
            thickness=2,
            font_scale=0.4,
            show=False,
            draw_score=False,
            out_file=out_file)

    def get_new_embeds(self, x, bboxes, img_meta, rescale=False):
        if rescale:
            scale_factor = img_meta[0]['scale_factor']
            bboxes = bboxes[:, :4] * scale_factor
        else:
            bboxes = bboxes[:, :4]
        rois = bbox2roi([bboxes])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        track_feats = self.asso_track_roi_extractor(
            x[:self.asso_track_roi_extractor.num_inputs], rois)
        bbox_embeds, _ = self.bbox_head.get_embeds(bbox_feats)
        track_embeds = self.asso_track_head.get_embeds(track_feats)
        return bbox_embeds, track_embeds
