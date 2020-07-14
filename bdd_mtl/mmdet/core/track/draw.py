import os
import mmcv
import subprocess
import numpy as np


def trans_results(inputs, num_classes):
    bboxes = []
    labels = []
    for x in inputs:
        if x is None:
            bboxes.append(None)
            labels.append(None)
            continue
        output = np.zeros((0, x[0].shape[1]))
        label = np.zeros((0, ), dtype=np.int)
        for cls_index in range(num_classes):
            y = x[cls_index]
            num_res = y.shape[0]
            if num_res > 0:
                output = np.concatenate((output, y), axis=0)
                _label = np.ones((num_res)) * cls_index
                label = np.concatenate((label, _label), axis=0)
        bboxes.append(output)
        labels.append(label)
    return bboxes, labels


def get_anns(img_id, dataset):
    ann_ids = dataset.bdd.getAnnIds([img_id])
    anns = dataset.bdd.loadAnns(ann_ids)
    ann_bboxes = np.zeros((0, 4))
    ann_labels = np.zeros((0, ), dtype=np.int)
    for ann in anns:
        box = ann['bbox']
        xyxy = np.array([[box[0], box[1], box[0] + box[2], box[1] + box[3]]])
        label = dataset.cat2label[ann['category_id']] - 1
        ann_bboxes = np.concatenate((ann_bboxes, xyxy), axis=0)
        ann_labels = np.concatenate((ann_labels, np.array([label])), axis=0)
    return ann_bboxes, ann_labels


def draw_results(results,
                 dataset,
                 out,
                 cfg,
                 score_thre=0.8,
                 plot=[False, True, False]):
    det_results = results['bbox_results']
    track_results = results['track_results']

    output = '{}_view'.format(out.split('.pkl')[0])
    if os.path.exists(output):
        print('Drawing output folder exists, rm it...')
        subprocess.call('rm -rf {}'.format(output), shell=True)
    os.makedirs(output, exist_ok=False)
    img_prefix = cfg.data.test.img_prefix
    num_classes = len(dataset.CLASSES)
    vid_ids = dataset.vid_ids
    count = 0
    for vid_id in vid_ids:
        print('---------------------')
        print('Drawing video {}...'.format(vid_id))
        img_ids = dataset.bdd.getImgIdsFromVideoId(vid_id)
        num_frames = len(img_ids)
        vid_bbox_results = det_results[count:count + num_frames]
        vid_track_results = track_results[count:count + num_frames]
        count += num_frames
        os.makedirs(os.path.join(output, str(vid_id)), exist_ok=True)
        bboxes, b_labels = trans_results(vid_bbox_results, num_classes)
        if plot[2]:
            tracked, t_labels = trans_results(vid_track_results, num_classes)
            for k in range(1, num_frames):
                if tracked[k].shape[0] > 0:
                    tracked[k] = np.concatenate(
                        (tracked[k], bboxes[k - 1][:, -1][:, None]), axis=1)
        for i, img_id in enumerate(img_ids):
            print('Drawing image {}'.format(img_id))
            img_name = dataset.bdd.loadImgs([img_id])[0]['file_name']
            # img current is a name!
            img = os.path.join(img_prefix, img_name)
            out_path = os.path.join(output, '{}/{}'.format(vid_id, img_name))
            if plot[0]:
                out = out_path if not (plot[1] or plot[2]) else None
                ann_bboxes, ann_labels = get_anns(img_id, dataset)
                img = mmcv.imshow_det_bboxes(
                    img,
                    ann_bboxes,
                    ann_labels,
                    bbox_color='green',
                    text_color='green',
                    show=False,
                    out_file=out)
            if plot[1]:
                out = out_path if not plot[2] else None
                img = mmcv.imshow_det_bboxes(
                    img,
                    bboxes[i],
                    b_labels[i],
                    score_thr=score_thre,
                    bbox_color='yellow',
                    text_color='yellow',
                    show=False,
                    out_file=out)
            if plot[2]:
                out = out_path
                if tracked[i] is None:
                    continue
                img = mmcv.imshow_det_bboxes(
                    img,
                    tracked[i],
                    t_labels[i],
                    score_thr=score_thre,
                    bbox_color='red',
                    text_color='red',
                    show=False,
                    out_file=out)
