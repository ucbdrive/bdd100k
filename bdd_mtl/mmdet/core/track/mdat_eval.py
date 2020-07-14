import motmetrics as mm
from collections import defaultdict
from pycocotools.bdd import BDD
from pycocotools.videoeval import VideoEval
import numpy as np


def xyxy2xywh(_bbox):
    # _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def mask_iou_matrix(objs, hyps, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis masks.
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    if len(objs) == 0 or len(hyps) == 0:
        return np.empty((0,0))
    # FIXME: add crowd attribute
    iscrowd = np.zeros(len(objs))
    # FIXME: remove height and weight
    objs = [maskUtils.frPyObjects(obj, 720, 1280) for obj in objs]
    objs = [maskUtils.merge(o) for o in objs]
    C = 1 - maskUtils.iou(hyps, objs, iscrowd)
    C[C > max_iou] = np.nan
    return C.transpose()

def get_track_ap(gt_file, results, with_occlusion=False, return_vis_info=False):
    # bdd ground truth and detection files
    bddGt = BDD(gt_file)
    bddDt = bddGt.loadRes(results)
    # prepare for returning for visualization
    if return_vis_info:
        out = dict()

    video_eval = VideoEval(bddGt, bddDt, iouType='bbox')
    # without and with re-ID
    video_eval.evaluate(with_occlusion=with_occlusion)
    video_eval.accumulate()
    video_eval.summarize()
    if return_vis_info:
        out['gt_tracks'] = video_eval._gts
        out['dt_tracks'] = video_eval._dts
        out['gt_matches'] = [evalVid['gtMatches'] if evalVid else None for evalVid in video_eval.evalVids]
        out['gt_ids'] = [evalVid['gtIds'] if evalVid else None for evalVid in video_eval.evalVids]
        out['dt_ids'] = [evalVid['dtIds'] if evalVid else None for evalVid in video_eval.evalVids]

    if return_vis_info:
        return out


def mdat_eval(all_results, dataset, out_file, cfg=None, ann_file=None, return_vis_info=False, with_mask=False):
    bdd = dataset.bdd
    img_infos = dataset.serial_img_infos
    assert len(all_results) == len(img_infos)
    track_acc = dict()
    mask_track_acc = dict()
    global_instance_id = 0
    num_instances = 0
    for cat_id in dataset.cat_ids:
        track_acc[cat_id] = mm.MOTAccumulator(auto_id=True)
    if with_mask:
        for cat_id in dataset.cat_ids:
            mask_track_acc[cat_id] = mm.MOTAccumulator(auto_id=True)

    for img_info, results in zip(img_infos, all_results):
        img_id = img_info['id']

        if img_info['first_frame']:
            global_instance_id += num_instances
        if len(list(results.keys())) > 0:
            num_instances = max(list(results.keys())) + 1

        pred_bboxes, pred_ids = defaultdict(list), defaultdict(list)
        if with_mask:
            pred_masks = defaultdict(list)
        for instance_id, result in results.items():
            _bbox = xyxy2xywh(result['bbox'])
            _cat = dataset.cat_ids[result['label']]
            pred_bboxes[_cat].append(_bbox)
            instance_id += global_instance_id
            pred_ids[_cat].append(instance_id)
            if with_mask:
                pred_masks[_cat].append(result['segmentation'])

        gt_bboxes, gt_ids = defaultdict(list), defaultdict(list)
        if with_mask:
            gt_masks = defaultdict(list)
        for cat_id in dataset.cat_ids:
            ann_ids = bdd.getAnnIds(img_id, catIds=[cat_id])
            anns = bdd.loadAnns(ann_ids)
            for ann in anns:
                gt_bboxes[cat_id].append(ann['bbox'])
                gt_ids[cat_id].append(ann['instance_id'])
                if with_mask:
                    gt_masks[cat_id].append(ann['segmentation'])
            distances = mm.distances.iou_matrix(
                gt_bboxes[cat_id], pred_bboxes[cat_id], max_iou=0.5)
            track_acc[cat_id].update(gt_ids[cat_id], pred_ids[cat_id],
                                     distances)
            if with_mask:
                mask_distances = mask_iou_matrix(
                    gt_masks[cat_id], pred_masks[cat_id], max_iou=0.5)
                mask_track_acc[cat_id].update(gt_ids[cat_id], pred_ids[cat_id],
                                         mask_distances)

    # eval for track
    print('Generating matchings and summary...')
    empty_cat = []
    for cat, v in track_acc.items():
        if len(v._events) == 0:
            empty_cat.append(cat)
    for cat in empty_cat:
        track_acc.pop(cat)

    print('Evaluating box tracking...')
    mh = mm.metrics.create()
    summary = mh.compute_many([v for cat, v in track_acc.items()],
                              # metrics=[
                              #     'mota', 'motp', 'num_misses',
                              #     'num_false_positives', 'num_switches',
                              #     'num_fragmentations', 'mostly_tracked',
                              #     'mostly_lost', 'partially_tracked', 're_id_success_rate'
                              # ],
                              metrics=[
                                  'mota', 'motp', 'num_matches', 'num_misses',
                                  'num_false_positives', 'num_switches'
                              ],
                              names=list(track_acc.keys()),
                              generate_overall=True)
    print('Printing...')
    summary['motp'] = (1 - summary['motp']) * 100
    summary['mota'] = summary['mota'] * 100
    print(summary)
    out = {k: v['OVERALL'] for k, v in summary.to_dict().items()}

    if with_mask:
        print('Evaluating seg tracking...')
        mh = mm.metrics.create()
        summary = mh.compute_many([v for cat, v in mask_track_acc.items()],
                                  # metrics=[
                                  #     'mota', 'motp', 'num_misses',
                                  #     'num_false_positives', 'num_switches',
                                  #     'num_fragmentations', 'mostly_tracked',
                                  #     'mostly_lost', 'partially_tracked', 're_id_success_rate'
                                  # ],
                                  metrics=[
                                      'mota', 'motp', 'num_matches', 'num_misses',
                                      'num_false_positives', 'num_switches'
                                  ],
                                  names=list(track_acc.keys()),
                                  generate_overall=True)
        print('Printing...')
        summary['motp'] = (1 - summary['motp']) * 100
        summary['mota'] = summary['mota'] * 100
        print(summary)

        out = {'seg_' + k: v['OVERALL'] for k, v in summary.to_dict().items()}
    return out
    # track AP
    # print('Evaluating Track AP...')
    # res = []
    # for img_info, results in zip(img_infos, all_results):
    #     for instance_id, result in results.items():
    #         res.append({
    #             'bbox': xyxy2xywh(result['bbox']),
    #             'score': result['bbox'][-1],
    #             'image_id': img_info['id'],
    #             'category_id': result['label'] + 1,
    #             'instance_id': instance_id + 1
    #         })
    # print('Without occlusion:')
    # # get_track_ap(cfg.data.test.ann_file, res)
    # get_track_ap(ann_file, res)
    # print('With occlusion:')
    # # get_track_ap(cfg.data.test.ann_file, res, with_occlusion=True)
    # get_track_ap(ann_file, res, with_occlusion=True)
