import torch


class SamplingResult(object):

    def __init__(self,
                 pos_inds,
                 neg_inds,
                 bboxes,
                 gt_bboxes,
                 assign_result,
                 gt_flags,
                 ref_bboxes=None):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        if assign_result.pids is not None:
            self.pos_gt_pids = assign_result.pids[pos_inds]
        else:
            self.pos_gt_pids = None

        if ref_bboxes is not None:
            self.pos_ref_bboxes = ref_bboxes[self.pos_gt_pids, :]
            no_match_ids = torch.nonzero(self.pos_gt_pids < 0)
            self.pos_ref_bboxes[no_match_ids, :] = torch.zeros(1, 4).cuda() - 1

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
