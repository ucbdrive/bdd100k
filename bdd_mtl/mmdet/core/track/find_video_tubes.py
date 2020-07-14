import torch
import numpy as np
from collections import defaultdict
from ..bbox.geometry import bbox_overlaps


def iou_score(v1, v2, binary=False):
    score = bbox_overlaps(torch.Tensor(v1[:, :4]),
                          torch.Tensor(v2[:, :4])).numpy()
    signal = (score > 0.5).astype(np.float32)
    if binary:
        return signal
    else:
        return score * signal


def scatter_by_classes(inputs, num_classes):
    outputs = defaultdict(list)
    for x in inputs:
        if x is None:
            for i in range(num_classes):
                outputs[i].append(None)
        else:
            for i, y in enumerate(x):
                outputs[i].append(y)
    return outputs


def gather_by_classes(inputs):
    outputs = []
    num_classes = len(inputs)
    num_frames = len(inputs[0])
    for i in range(num_frames):
        per_frame = []
        for j in range(num_classes):
            per_frame.append(inputs[j][i])
        outputs.append(per_frame)
    return outputs


def det_to_track(det_results):
    num_classes = len(det_results[0])
    outputs = []
    for i, bboxes in enumerate(det_results):
        output = {}
        for j in range(num_classes):
            results = bboxes[j]
            num_bboxes = results.shape[0]
            for k in range(num_bboxes):
                _box = results[k, :]
                instance_id = _box[-1]
                if instance_id >= 0:
                    output[int(instance_id)] = {'bbox': _box[:-1], 'label': j}
        outputs.append(output)
    return outputs


def check_diff(input1, input2):
    for x, y in zip(input1, input2):
        for _x, _y in zip(x, y):
            score_diff = _x[:, 4] - _y[:, 4]
            for score in score_diff:
                if score > 0.1:
                    print(score)


def finding_video_tubes(outputs, dataset):
    """Implementation of `Linking tracklets to object tubes` in the Paper
    "Detect to Track and Track to Detect."

    Inputs:
        track_results (list): -> dict() -> keys:
                                        ['bbox_results', 'track_results']
        dataset: CLASS DATASET
    """
    num_classes = len(dataset.CLASSES)
    all_bbox_results = outputs['bbox_results']
    all_track_results = outputs['track_results']
    all_outputs = defaultdict(list)

    count = 0
    instance_id = 0
    vid_ids = dataset.vid_ids
    for vid_id in vid_ids:
        vid_name = dataset.bdd.loadVideos(vid_id)
        print(vid_name)
        img_ids = dataset.bdd.getImgIdsFromVideoId(vid_id)
        num_frames = len(img_ids)
        vid_bbox_results = all_bbox_results[count:count + num_frames]
        vid_track_results = all_track_results[count:count + num_frames]
        count += num_frames
        assert vid_track_results[0] is None, 'Maybe split videos incorrectly.'
        class_bbox_results = scatter_by_classes(vid_bbox_results, num_classes)
        class_track_results = scatter_by_classes(vid_track_results,
                                                 num_classes)
        outputs = []
        for cls_index in range(num_classes):
            det, instance_id = finding_video_tubes_greedy_per_class(
                class_bbox_results[cls_index], class_track_results[cls_index],
                instance_id, cls_index)
            outputs.append(det)
        track_results = track_gather(outputs, img_ids)
        # bbox_results = gather_by_classes(outputs)
        # check_diff(bbox_results, vid_bbox_results)
        # track_results = det_to_track(bbox_results)
        # all_outputs['bbox_results'].extend(bbox_results)
        all_outputs['track_results'].extend(track_results)

    return all_outputs


def track_gather(outputs, img_ids):
    num_frames = len(img_ids)
    output_list = []
    for k in range(num_frames):
        out = defaultdict(list)
        for res in outputs:
            if k in res.keys():
                out.update(res[k])
        output_list.append(out)
    return output_list


def finding_video_tubes_viterbi_per_class(bbox_results, track_results, path_id,
                                          cls_index):
    ori_bboxes = bbox_results.copy()

    empty_set = False
    for i, _ori_bboxes in enumerate(ori_bboxes):
        num_bboxes = _ori_bboxes.shape[0]
        if num_bboxes == 0:
            empty_set = True
        ids = np.zeros((num_bboxes, 1)) - 1
        ori_bboxes[i] = np.concatenate((ori_bboxes[i], ids), axis=1)
    if empty_set:
        return ori_bboxes, path_id

    num_frames = len(bbox_results)
    vertices_isempty = np.zeros(num_frames, dtype=np.bool)

    paths = []
    while not np.any(vertices_isempty):
        data_score = []
        data_idx = []
        for i in range(num_frames):
            num_bboxes = bbox_results[i].shape[0]
            data_score.append(np.zeros((num_bboxes, 1)))
            data_idx.append(np.zeros((num_bboxes, 1)))
            data_idx[i][:] = np.nan

        # for i in range(1, num_frames):
        #     track_results[i] = np.concatenate(
        #         (track_results[i], bbox_results[i - 1][:, -1][:, None]),
        #         axis=1)
        #     edge_score = iou_score(track_results[i],
        #                            bbox_results[i])  #[N_t-1, N_t]
        #     # TODO: why this
        #     # edge_score += np.transpose(data_score[i])
        #     edge_score += bbox_results[i][:, 4][None, :]
        #     edge_score += track_results[i][:, 4][:, None]
        #     data_score[i] = np.max(edge_score, axis=1)
        #     data_idx[i] = np.argmax(edge_score, axis=1)
        for i in range(num_frames - 1, 0, -1):
            track_results[i] = np.concatenate(
                (track_results[i], bbox_results[i - 1][:, -1][:, None]),
                axis=1)
            edge_score = iou_score(track_results[i],
                                   bbox_results[i])  # [N_t-1, N_t]
            if i < num_frames - 2:
                edge_score += np.transpose(data_score[i + 1])
            edge_score += bbox_results[i][:, 4][None, :]
            edge_score += track_results[i][:, 4][:, None]
            data_score[i] = np.max(edge_score, axis=1)
            data_idx[i] = np.argmax(edge_score, axis=1)

        box_index = np.argmax(data_score[1])
        boxes = bbox_results[0][box_index, :4]
        index = np.array(box_index)
        scores = np.array(bbox_results[0][box_index, 4])
        for i in range(1, num_frames):
            box_index = data_idx[i][box_index]
            index = np.hstack((index, np.array(box_index)))
            boxes = np.vstack((boxes, bbox_results[i][box_index, :4]))
            scores = np.hstack(
                (scores, np.array(bbox_results[i][box_index, 4])))
        cur_list = [index, boxes, scores]
        paths.append(cur_list)

        for i in range(num_frames):
            mask = np.ones(bbox_results[i].shape[0], dtype=np.bool)
            mask[index[i]] = False
            bbox_results[i] = bbox_results[i][mask, :]
            if not i == num_frames - 1:
                track_results[i + 1] = track_results[i + 1][mask, :]
            vertices_isempty[i] = (bbox_results[i].shape[0] == 0)

    unmap_idx_list = []
    for i in range(num_frames):
        trimmed_idx = [path[0][i] for path in paths]
        flag = np.zeros(ori_bboxes[i].shape[0])
        ori_idx = []
        for j in trimmed_idx:
            count = -1
            for k in range(ori_bboxes[i].shape[0]):
                if not flag[k]:
                    count += 1
                    if count == j:
                        ori_idx.append(k)
                        flag[k] = True
        unmap_idx_list.append(ori_idx)

    for cur_path_id, path in enumerate(paths):
        path_score = path[2]
        path_score_t = sorted(path_score)[len(path_score) // 2:]
        ave_score = sum(path_score_t) / len(path_score_t)
        for i in range(num_frames):
            unmap_idx = unmap_idx_list[i][cur_path_id]
            ori_bboxes[i][unmap_idx, 5] = path_id
            # score = ori_bboxes[i][unmap_idx, 4]
            # if score < ave_score:
            ori_bboxes[i][unmap_idx, 4] += ave_score
            ori_bboxes[i][unmap_idx, 4] /= 2
        path_id += 1

    return ori_bboxes, path_id


def finding_video_tubes_greedy_per_class(bbox_results, track_results, path_id,
                                         cls_index):
    # ori_bboxes = bbox_results.copy()
    #
    # empty_set = False
    # for i, _ori_bboxes in enumerate(ori_bboxes):
    #     num_bboxes = _ori_bboxes.shape[0]
    #     if num_bboxes == 0:
    #         empty_set = True
    #     ids = np.zeros((num_bboxes, 1)) - 1
    #     ori_bboxes[i] = np.concatenate((ori_bboxes[i], ids), axis=1)
    # if empty_set:
    #     return None, path_id

    num_frames = len(bbox_results)
    paths = []

    for i in range(1, num_frames):
        track_results[i] = np.concatenate(
            (track_results[i], bbox_results[i - 1][:, -1][:, None]), axis=1)

    # per frame calculation
    for start_t in range(num_frames - 1):
        num_iters = bbox_results[start_t].shape[0]
        _iter = 0
        while _iter < num_iters:
            data_score = []
            data_idx = []

            for i in range(start_t, num_frames):
                num_bboxes = bbox_results[i].shape[0]
                data_score.append(np.zeros((num_bboxes, 1)))
                data_idx.append(np.zeros((num_bboxes, 1)))
                data_idx[i - start_t][:] = np.nan

            for i in range(start_t + 1, num_frames):
                edge_score = iou_score(track_results[i],
                                       bbox_results[i])  # [N_t-1, N_t]
                # TODO: why this
                # edge_score += np.transpose(data_score[i])
                # edge_score += bbox_results[i][:, 4][None, :]
                # edge_score += track_results[i][:, 4][:, None]
                if (edge_score.shape[0]) > 0 and (edge_score.shape[1] > 0):
                    data_score[i - start_t] = np.max(edge_score, axis=1)
                    data_idx[i - start_t] = np.argmax(edge_score, axis=1)

            if len(data_score[1]) == 0:
                _iter += 1
                continue
            box_index = np.argmax(data_score[1])
            boxes = bbox_results[start_t][box_index, :4]
            index = np.array(box_index)
            scores = np.array(bbox_results[start_t][box_index, 4])
            for i in range(1, num_frames - start_t):
                if len(data_score[i]) == 0:
                    break
                iou = data_score[i][box_index]
                if iou > 0.75:
                    box_index = data_idx[i][box_index]
                    index = np.hstack((index, np.array(box_index)))
                    boxes = np.vstack(
                        (boxes, bbox_results[start_t + i][box_index, :4]))
                    scores = np.hstack(
                        (scores,
                         np.array(bbox_results[start_t + i][box_index, 4])))
                else:
                    break
            cur_list = [index, boxes, scores, start_t]
            end_i = i
            if cur_list[0].size > 1 and len(cur_list[0]) > 1:
                paths.append(cur_list)
                for i in range(start_t, start_t + end_i):
                    mask = np.ones(bbox_results[i].shape[0], dtype=np.bool)
                    mask[index[i - start_t]] = False
                    bbox_results[i] = bbox_results[i][mask, :]
                    if not i == num_frames - 1:
                        track_results[i + 1] = track_results[i + 1][mask, :]
            _iter += 1

    tracklets = defaultdict(list)
    for path in paths:
        start_t = path[-1]
        max_score = path[2].max()
        if max_score < 0.8:
            continue
        for k in range(path[1].shape[0]):
            _bbox = np.append(path[1][k, :], max_score)
            if start_t + k not in tracklets.keys():
                tracklets[start_t + k] = defaultdict(list)
            tracklets[start_t + k][path_id] = dict(bbox=_bbox, label=cls_index)
        path_id += 1

    #
    # unmap_idx_list = []{}
    # for i in range(start_t, num_frames):
    #     trimmed_idx = [path[0][i] for path in paths]
    #     flag = np.zeros(ori_bboxes[i].shape[0])
    #     ori_idx = []
    #     for j in trimmed_idx:
    #         count = -1
    #         for k in range(ori_bboxes[i].shape[0]):
    #             if not flag[k]:
    #                 count += 1
    #                 if count == j:
    #                     ori_idx.append(k)
    #                     flag[k] = True
    #     unmap_idx_list.append(ori_idx)
    #
    # for cur_path_id, path in enumerate(paths):
    #     path_score = path[2]
    #     path_score_t = sorted(path_score)[len(path_score) // 2:]
    #     ave_score = sum(path_score_t) / len(path_score_t)
    #     for i in range(num_frames):
    #         unmap_idx = unmap_idx_list[i][cur_path_id]
    #         ori_bboxes[i][unmap_idx, 5] = path_id
    #         score = ori_bboxes[i][unmap_idx, 4]
    #         # if score < ave_score:
    #         ori_bboxes[i][unmap_idx, 4] += ave_score
    #         ori_bboxes[i][unmap_idx, 4] /= 2
    #     path_id += 1

    return tracklets, path_id
