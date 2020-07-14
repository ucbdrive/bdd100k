import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import HEADS
from mmdet.core import bbox_overlaps


@HEADS.register_module
class AffinityHead(nn.Module):

    def __init__(self, affinity='overlap'):
        super(AffinityHead, self).__init__()
        assert affinity in ['overlap', 'similarity', 'all']
        self.affinity = affinity

    def init_weights(self):
        pass

    def test_forward(self, proposals, prop_bboxes, asso_probs, cfg,
                     vanish_frames, prev_ids):
        # valid_tracklet_idxs, append 0 for new objects as the begining.
        valid_t_idxs = torch.nonzero(
            vanish_frames < cfg.long_term_frames).squeeze(1) + 1
        valid_t_idxs = torch.cat(
            (torch.tensor([0], dtype=torch.long,
                          device=valid_t_idxs.device), valid_t_idxs))
        # Similarity with Softmax
        asso_scores = torch.zeros_like(asso_probs)
        asso_scores[:, valid_t_idxs] = F.softmax(
            asso_probs[:, valid_t_idxs], dim=1)  # [N_det, N_emb + 1]
        asso_scores *= (asso_scores > cfg.asso_score_thre).float()
        # Overlaps
        overlaps = torch.zeros_like(asso_scores)
        valid_prop = prop_bboxes[:, -1] > cfg.prop_score_thre
        prop_bboxes = prop_bboxes[valid_prop, :]
        prev_ids = prev_ids[valid_prop]
        prop_overlaps = bbox_overlaps(proposals[:, :4], prop_bboxes[:, :4])
        overlaps[:, prev_ids + 1] = prop_overlaps
        overlaps *= (overlaps > cfg.prop_overlap_thre).float()

        if self.affinity == 'overlap':
            return overlaps
        elif self.affinity == 'similarity':
            return asso_scores
        elif self.affinity == 'all':
            return overlaps + asso_scores

    def forward(self,
                proposals,
                prop_bboxes,
                asso_probs,
                cfg,
                vanish_frames=None,
                prev_ids=None,
                sampling_results=None):
        if vanish_frames is not None:
            affinity = self.test_forward(proposals, prop_bboxes, asso_probs,
                                         cfg, vanish_frames, prev_ids)
        else:
            affinity = self.train_forward(proposals, prop_bboxes, asso_probs,
                                          cfg)
        return affinity
