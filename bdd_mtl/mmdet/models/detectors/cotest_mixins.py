import torch
import torch.nn.functional as F
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms,
                        track2results, bbox2result, bbox_overlaps,
                        max_matching)
import numpy as np


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head.naive_forward(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class TrackTestMixin(object):

    def setup_tracklets(self,
                        x,
                        det_bboxes,
                        det_labels,
                        img_meta,
                        rescale=False,
                        cfg=None):
        # if current frame is first frame, init a tracklets sequence
        # filter bounding boxes with a pre-defined threshold
        valid_idx = det_bboxes[:, 4] > self.test_cfg.track.new_obj_score_thre
        valid_bboxes = det_bboxes[valid_idx]
        valid_labels = det_labels[valid_idx]
        bbox_embeds, track_embeds = self.get_new_embeds(
            x, valid_bboxes, img_meta, rescale)
        ids = torch.arange(valid_bboxes.shape[0]).to(track_embeds.device)
        vanish_frames = torch.zeros_like(valid_labels)
        self.update(
            type='inplace',
            embeddings=track_embeds,
            bbox_embeds=bbox_embeds,
            tracklet_scores=valid_bboxes[:, 4].detach().clone(),
            bboxes=valid_bboxes.detach().clone(),
            labels=valid_labels,
            vanish_frames=vanish_frames,
            prev_feats=x,
            prev_ids=ids)
        track_results = track2results(valid_bboxes, valid_labels, ids)
        return track_results

    def simple_test_prop_track(self, x, img_meta, rescale=False):
        assert img_meta[0][
            'first_frame'] is False, 'First frame cannot do tracking.'
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        prev_feats = self.prev_feats
        prev_bboxes = self.bboxes[self.prev_ids, :]
        if rescale:
            prev_bboxes[:, :4] *= scale_factor
        track_x = self.extract_corr_feat(prev_feats, x)
        prop_rois = bbox2roi([prev_bboxes])
        prop_track_feats = self.prop_track_roi_extractor(
            track_x[:self.prop_track_roi_extractor.num_inputs], prop_rois)
        if prop_track_feats.nelement() == 0:
            return torch.zeros(0, 5).to(prop_track_feats.device)
        prop_prob, prop_loc = self.prop_track_head(prop_track_feats)
        prop_bboxes = self.prop_track_head.get_track_bboxes(
            prop_rois,
            prop_prob,
            prop_loc,
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None)
        return prop_bboxes

    def simple_test_asso_track(self, x, proposals):
        # calculate associate props
        rois = bbox2roi(proposals)
        feats = self.asso_track_roi_extractor(
            x[:self.asso_track_roi_extractor.num_inputs], rois)
        track_embeds = self.asso_track_head.get_embeds(feats)
        asso_probs = self.asso_track_head.get_similarity(
            track_embeds, self.embeddings)
        return asso_probs

    def simple_test_asso_track_old(self,
                                   x,
                                   det_bboxes,
                                   img_meta,
                                   rescale=False):
        scale_factor = img_meta[0]['scale_factor']
        if rescale:
            det_bboxes[:, :4] *= scale_factor

        # calculate associate props
        rois = bbox2roi([det_bboxes])
        feats = self.asso_track_roi_extractor(
            x[:self.asso_track_roi_extractor.num_inputs], rois)
        cur_embs = self.asso_track_head.get_embeds(feats)
        self.cur_embs = cur_embs
        asso_probs = self.asso_track_head.get_similarity(
            cur_embs, self.embeddings)
        return asso_probs

    def simple_syner_test(self,
                          x,
                          proposals,
                          affinity,
                          img_meta,
                          cfg,
                          rescale=False):
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred, ids = self.bbox_head(
            roi_feats, self.bbox_embeds, affinity, extract_ref=False)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels, ids = self.bbox_head.get_det_and_ids(
            rois,
            cls_score,
            bbox_pred,
            ids,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=cfg)
        return det_bboxes, det_labels, ids

    def assign_ids(self, x, prop_bboxes, asso_probs, det_bboxes, det_labels,
                   img_meta, rescale):
        """Integrate matching score through
            1. Association score with softmax
            2. Semantic consistence
            3. Spatial overlap (Only in consective frames)
            4. Detection confidence
        """
        cfg = self.test_cfg.track
        # id init
        ids = torch.zeros_like(det_bboxes[:, 1]).long() - 1
        # Get semantic consistence, as a flag for assigning ids.
        cat_same = (self.labels == det_labels.view(-1, 1)).float()
        cat_dummy = cat_same.new_ones(cat_same.size(0), 1)
        cat_same = torch.cat((cat_dummy, cat_same), dim=1)
        # calculate feature appearance similarity
        valid_t_idxs = torch.nonzero(
            self.vanish_frames < cfg.long_term_frames).squeeze(1) + 1
        valid_t_idxs = torch.cat(
            (torch.tensor([0], dtype=torch.long,
                          device=valid_t_idxs.device), valid_t_idxs))
        if asso_probs.size(-1) < cat_same.size(-1):
            # Sigmoid or Cosine
            assert asso_probs.size(-1) + 1 == cat_same.size(-1)
            asso_scores = torch.zeros_like(cat_same)
            if asso_scores.max() > 1.0:
                asso_scores[:, 1:] = torch.sigmoid(asso_probs)
            asso_scores[:, 0] += 0.5
        else:
            # Softmax
            assert asso_probs.size(-1) == cat_same.size(-1)
            asso_scores = torch.zeros_like(cat_same)
            asso_scores[:, valid_t_idxs] = F.softmax(
                asso_probs[:, valid_t_idxs], dim=1)  # [N_det, N_emb + 1]
            asso_scores *= (asso_scores > cfg.asso_score_thre).float()
        # get overlaps under short-term tracking
        overlaps = torch.zeros_like(cat_same)
        valid_prop = prop_bboxes[:, -1] > cfg.prop_score_thre
        prop_bboxes = prop_bboxes[valid_prop, :]
        prev_ids = self.prev_ids[valid_prop]
        prop_overlaps = bbox_overlaps(det_bboxes[:, :4], prop_bboxes[:, :4])
        overlaps[:, prev_ids + 1] = prop_overlaps
        overlaps *= (overlaps > cfg.prop_overlap_thre).float()
        # short-term matching according to overlaps
        short_scores = overlaps * cat_same
        if cfg.clean_before_short_assign:
            valid_dets = det_bboxes[:, -1] > cfg.new_obj_score_thre
            valid_dets = valid_dets.view(-1, 1).repeat(1, short_scores.size(1))
            short_scores = short_scores * valid_dets.float()
        if prev_ids.shape[0] > 0 and (short_scores > 0).any():
            t2d_idxs = max_matching(short_scores[:, prev_ids + 1])
            is_match = t2d_idxs >= 0
            t2d_idxs = t2d_idxs[is_match]
            prev_ids = prev_ids[is_match]
            ids[t2d_idxs.tolist()] = prev_ids
            if cfg.prop_fn:
                raise NotImplementedError
        if cfg.use_reid:
            # long-term associtation
            valid_dets = ids < 0
            if cfg.clean_before_long_assign:
                valid_dets *= det_bboxes[:, -1] > cfg.new_obj_score_thre
            long_scores = asso_scores * cat_same
            valid_embeds = self.vanish_frames < cfg.long_term_frames
            if prev_ids.shape[0] > 0:
                valid_embeds[prev_ids] = 0
            long_scores[:, 1:] *= valid_embeds.float().view(-1, 1).repeat(
                1, long_scores.size(0)).transpose(1, 0)
            if valid_dets.any() and (long_scores[valid_dets, :] > 0).any():
                valid_d_idxs = torch.nonzero(valid_dets == 1).squeeze(1)
                d2t_idxs = max_matching(long_scores[valid_d_idxs, :].t()) - 1
                is_match = d2t_idxs >= 0
                ids[valid_d_idxs[is_match]] = d2t_idxs[is_match]
        # new objects
        valid_dets = ids < 0
        valid_dets *= det_bboxes[:, -1] > cfg.new_obj_score_thre
        valid_idxs = torch.nonzero(valid_dets > 0).squeeze(1).tolist()
        if len(valid_idxs) > 0:
            for i, valid_idx in enumerate(valid_idxs):
                ids[valid_idx] = self.embeddings.size(0) + i
            new_track_bboxes = det_bboxes[valid_idxs, :]
            new_track_labels = det_labels[valid_idxs]
            vanish_frames = torch.zeros_like(new_track_labels)
            bbox_embeds, track_embeds = self.get_new_embeds(
                x, new_track_bboxes, img_meta, rescale)
            self.update(
                type='contact',
                embeddings=track_embeds,
                bbox_embeds=bbox_embeds,
                tracklet_scores=new_track_bboxes[:, -1].detach().clone(),
                bboxes=new_track_bboxes.detach().clone(),
                labels=new_track_labels,
                vanish_frames=vanish_frames)
        return ids

    def update_tracklets(self, x, ids, det_bboxes, det_labels, img_meta,
                         rescale):
        """
            1. Update overall tracklets
            2. Update new_det_bboxes
            3. Update track_bboxes
        """
        match_idx = torch.nonzero(ids >= 0).squeeze(1).tolist()
        instance_ids = ids[match_idx].tolist()
        if len(match_idx) > 0:
            bbox_embeds, track_embeds = self.get_new_embeds(
                x, det_bboxes[match_idx], img_meta, rescale)
            # update embedding
            m = self.test_cfg.track.embed_momentum
            self.embeddings[instance_ids] = (1 - m) * self.embeddings[
                instance_ids].detach().clone() + m * track_embeds
            self.bbox_embeds[instance_ids] = (1 - m) * self.bbox_embeds[
                instance_ids].detach().clone() + m * bbox_embeds
        # TODO better way to update tracklet_scores
        self.tracklet_scores[instance_ids] = torch.max(
            self.tracklet_scores[instance_ids].detach().clone(),
            det_bboxes[match_idx, -1].detach().clone())
        # assert for label
        assert torch.equal(self.labels[instance_ids],
                           det_labels[match_idx])
        # update vanish frames
        update_vanish = torch.ones_like(self.vanish_frames)
        matched_ids = ids[ids >= 0].tolist()
        update_vanish[matched_ids] -= 1
        self.vanish_frames += update_vanish
        # update bboxes
        self.bboxes[instance_ids] = det_bboxes[match_idx].detach().clone()
        # generate new tracklets
        track_bboxes = det_bboxes[match_idx, :]
        track_labels = det_labels[match_idx]
        ids = ids[match_idx]
        track_results = track2results(track_bboxes, track_labels, ids)
        self.update(type='inplace', prev_feats=x, prev_ids=ids)
        return track_results


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
