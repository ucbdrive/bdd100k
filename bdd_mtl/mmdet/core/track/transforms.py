import torch
import numpy as np
from collections import defaultdict


def track2results(bboxes, labels, ids):
    bboxes = bboxes.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    ids = ids.cpu().numpy().tolist()
    outputs = defaultdict(list)
    for bbox, label, id in zip(bboxes, labels, ids):
        outputs[id] = dict(bbox=bbox, label=label)
    return outputs

def segtrack2results(bboxes, masks, labels, ids):
    labels = labels.cpu().numpy().tolist()
    bboxes = bboxes.cpu().numpy().tolist()
    ids = ids.cpu().numpy().tolist()
    outputs = defaultdict(list)
    for bbox, mask, label, id in zip(bboxes, masks, labels, ids):
        outputs[id] = dict(segmentation=mask, bbox=bbox, label=label)
    return outputs


def results2track(track_results):
    track_bboxes = np.zeros((0, 5))
    track_ids = np.zeros((0))
    for k, v in track_results.items():
        track_bboxes = np.concatenate((track_bboxes, v['bbox'][None, :]),
                                      axis=0)
        track_ids = np.concatenate((track_ids, np.array([k])), axis=0)
    return track_bboxes, track_ids


def max_matching(scores):
    """ Matching according to maximum similarity.
        Input matrix shape: [D, T] (dim=0)
        Return ids shape: [T]
    """
    ids = scores.new_full((scores.size(1), ), -1).long()
    # assume t2d
    t2d_max, t2d_idxs = scores.max(dim=0)
    match_t_idxs = torch.nonzero(t2d_max > 0).squeeze(1)
    if match_t_idxs.shape[0] == 0:
        return ids
    t2d_max = t2d_max[match_t_idxs.tolist()]
    t2d_idxs = t2d_idxs[match_t_idxs.tolist()]
    _, d2t_idxs = scores[:, match_t_idxs].max(dim=1)
    flag_d2t = d2t_idxs[t2d_idxs].long()
    flag_t2d = torch.arange(t2d_idxs.size(0)).to(scores.device)
    is_match = (flag_d2t == flag_t2d)
    if not is_match.all():
        true_match_t_idxs = torch.nonzero(is_match == 1).squeeze(1)
        ids[match_t_idxs[true_match_t_idxs]] = t2d_idxs[true_match_t_idxs]
        scores[t2d_idxs[true_match_t_idxs], :] = 0
        unmatch_t_idxs = torch.nonzero(is_match == 0).squeeze(1)
        _ids = max_matching(scores[:, match_t_idxs[unmatch_t_idxs]])
        ids[match_t_idxs[unmatch_t_idxs]] = _ids
    else:
        ids[match_t_idxs] = t2d_idxs
    return ids
