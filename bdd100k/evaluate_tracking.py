from collections import defaultdict
import time
import importlib
import os
import motmetrics as mm
import numpy as np
from motmetrics.lap import linear_sum_assignment
import sys
import json
import time
import argparse

def xyxy2xywh(bbox):
    return [
        bbox['x1'],
        bbox['y1'],
        bbox['x2'] - bbox['x1'],
        bbox['y2'] - bbox['y1']
    ]


def intersection_over_area(preds, gts):
    """
    Returns the intersection over the area of the predicted box
    """
    out = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            x1, x2 = max(p[0], g[0]), min(p[0] + p[2], g[0] + g[2])
            y1, y2 = max(p[1], g[1]), min(p[1] + p[3], g[1] + g[3])
            out[i][j] = max(x2 - x1, 0) * max(y2 - y1, 0) / float(p[2] * p[3])
        
    return out



def preprocessResult(all_results, anns, crowd_ioa_thr=0.5):
    """Preprocesses data
    Returns a subset of the predictions.
    """
    # pylint: disable=too-many-locals

    dropped_gt_ids = set()
    dropped_gts = []
    drops = 0
    num_preds = sum([len(i) for i in all_results.values()])
    # match
    for img in anns:
        res = all_results[img['name']]
        gt_bboxes = [xyxy2xywh(a['box2d']) for a in img['labels'] if not a['attributes']['Crowd']]
        res_bboxes = [xyxy2xywh(l['box2d']) for l in res]
        dropped_pred = set()
        # drop preds that match with ignored labels
        dist = mm.distances.iou_matrix(gt_bboxes, res_bboxes, max_iou=0.5)
        le, ri = linear_sum_assignment(dist)

        ignore_gt = [a['category'] in ignored_cats for a in img['labels'] if not a['attributes']['Crowd']]
        fp_ids = set(range(len(res_bboxes)))
        for i, j in zip(le, ri):
            if not np.isfinite(dist[i, j]):
                continue
            fp_ids.remove(j)
            if ignore_gt[i]:
                # remove from results
                dropped_gt_ids.add(img['labels'][i]['id'])
                dropped_pred.add(j)
                dropped_gts.append(img['labels'][i])

        # drop fps that fall in crowd regions
        crowd_gt_labels = [xyxy2xywh(a['box2d']) for a in img['labels'] if a['attributes']['Crowd']]

        if len(crowd_gt_labels) > 0 and len(fp_ids) > 0:
            ioas = np.max(intersection_over_area([res_bboxes[i] for i in fp_ids], crowd_gt_labels), axis=1)
            for i, ioa in zip(fp_ids, ioas):
                if ioa > crowd_ioa_thr:
                    dropped_pred.add(i)

        all_results[img['name']] = [r for i, r in enumerate(res) if not i in dropped_pred]

    print('Ignored {} detected boxes.'.format(num_preds - sum([len(i) for i in all_results.values()])))


def mmeval(anns, all_results, cats_mapping=None):

    assert len(all_results) == len(anns)
    
    track_acc = dict()
    mask_track_acc = dict()
    global_instance_id = 0
    global_instance_dict = dict()
    video_id = -1
    cat_ids = np.unique(list(cats_mapping.values()))
    for cat_id in cat_ids:
        track_acc[cat_id] = mm.MOTAccumulator(auto_id=True)

    for img in anns:
        img['labels'] = [a for a in img['labels'] if not ((a['category'] in ignored_cats) or (a['attributes']['Crowd']))]
        if img['index'] == 0:
            video_id += 1
        results = all_results[img['name']]

        pred_bboxes, pred_ids = defaultdict(list), defaultdict(list)
        for pred in results:
            key = (video_id, pred['id'])
            if not key in global_instance_dict:
                global_instance_dict[key] = global_instance_id
                global_instance_id += 1
            _bbox = xyxy2xywh(pred['box2d'])
            _cat = cats_mapping[pred['category']]
            pred_bboxes[_cat].append(_bbox)
            instance_id = global_instance_dict[key]
            pred_ids[_cat].append(instance_id)

        gt_bboxes, gt_ids = defaultdict(list), defaultdict(list)

        for ann in img['labels']:
            _cat = cats_mapping[ann['category']]
            gt_bboxes[_cat].append(xyxy2xywh(ann['box2d']))
            gt_ids[_cat].append(ann['id'])
        
        for cat_id in cat_ids:
            distances = mm.distances.iou_matrix(
                gt_bboxes[cat_id], pred_bboxes[cat_id], max_iou=0.5)
            track_acc[cat_id].update(gt_ids[cat_id], pred_ids[cat_id],
                                         distances)

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
                              metrics=[
                                  'mota', 'motp', 'num_misses',
                                  'num_false_positives', 'num_switches',
                                  'mostly_tracked',
                                  'mostly_lost', 'partially_tracked'
                              ],
                              names=list(track_acc.keys()),
                              generate_overall=True)
    print('Printing...')
    summary['motp'] = (1 - summary['motp']) * 100
    summary['mota'] = summary['mota'] * 100

    print(summary)
    out = {k: v for k, v in summary.to_dict().items()}
    return out


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument(
          "-a", "--ann-dir",
          default="/annotation/directory/",
          help="path to tracking annotation base folder",
    )
    parser.add_argument(
          "-r", "--res-json",
          default="/result/json",
          help="path to prediction result json",
    )
    parser.add_argument(
          "-o", "--out-dir",
          default=".",
          help="directory to save output scores",
    )
    return parser.parse_args()


if __name__ == '__main__':

    super_category_map = {
        'pedestrian': 'person',
        'rider': 'person',
        'car': 'vehicle',
        'bus': 'vehicle',
        'truck': 'vehicle',
        'train': 'vehicle',
        'motorcycle': 'bike',
        'bicycle': 'bike',
        'trailer': 'ignored',
        'other person': 'ignored',
        'other vehicle': 'ignored'
    }
    ignored_cats = ['trailer', 'other person', 'other vehicle']

    args = parse_arguments()
    tic = time.time()

    # # ====================================
    # # local debug
    # # ====================================

    # set up resFile
    if not os.path.exists(args.res_json):
        raise Exception("%s doesn't exist"%(args.res_json))

    with open(args.res_json) as data_file:    
        _res = json.load(data_file)

    # fast indexing
    res = defaultdict(list)
    for r in _res:
        res[r['name']] = r['labels']

    anns = []
    for fn in os.listdir(args.ann_dir):
        with open(os.path.join(args.ann_dir, fn)) as f:
            anns += json.load(f)

    scores = []

    # ignore predictions
    preprocessResult(res, anns)

    # tracking evaluation
    eval_res = mmeval(anns, res, cats_mapping={k: k for k in super_category_map.keys()})
    [scores.append('{}_{}: {}'.format(k1, k2, v)) for k1 in eval_res.keys() for k2, v in eval_res[k1].items()]
    # average by class
    scores.append('mmota: {}'.format(np.mean([v for k, v in eval_res['mota'].items() if k != 'OVERALL' and v == v])))
    scores.append('mmotp: {}'.format(np.mean([v for k, v in eval_res['motp'].items() if k != 'OVERALL' and v == v])))
    eval_res_super = mmeval(anns, res, cats_mapping=super_category_map)
    [scores.append('{}_{}: {}'.format(k1, k2, v)) for k1 in eval_res_super.keys() for k2, v in eval_res_super[k1].items() if not k2 == 'OVERALL']

    output_filename = os.path.join(outDir, 'scores.txt')
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(scores))
    output_file.flush()
    output_file.close()

    toc = time.time()
    print('Done (t=%0.2fs)' % (toc-tic))