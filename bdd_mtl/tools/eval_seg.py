import argparse
import boundary_utils as bu
import numpy as np
import os
import sys
import time
from utils import *
from multiprocessing import Pool
import pickle as pk
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate drivable area and semantic segmentation predictions')
    parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('-p', '--pred-dir', default=None)
    args = parser.parse_args()
    return args

def _eval_drivable(infos):
    global task
    gt_fn, pred_fn = infos
    gt = np.array(Image.open(gt_fn))

    pred = np.load(pred_fn)
    drivable_hist = fast_hist(pred.flatten(), gt.flatten(), 3)

    return [drivable_hist]

def _eval_sem_seg(infos):
    global task
    gt_fn, pred_fn = infos
    gt = np.array(Image.open(gt_fn))
    pred = np.load(pred_fn).squeeze(0).astype(np.uint8)

    # semantic segmentation
    hist = fast_hist(pred.flatten(), gt.flatten(), 19)

    return hist

def main():

    args = parse_args()

    tasks = os.listdir(args.pred_dir)
    # segmentation

    if 'sem_seg' in tasks:
        print('Evaluating semantic segmentation...')
        sem_seg_base = os.path.join(args.data_dir, 'images', '10k', 'val')
        gt_fns = [os.path.join(args.data_dir, 'labels', 'sem_seg', 'sem_seg_val', fn[:-4] + '_train_id.png') for fn in os.listdir(sem_seg_base)]
        sem_seg_fns = [os.path.join(args.pred_dir, 'sem_seg', '{}.npy'.format(fn[:-4])) for fn in os.listdir(sem_seg_base)]
        pool = Pool(5)
        o = pool.imap_unordered(_eval_sem_seg, zip(gt_fns, sem_seg_fns))
        tic = time.time()
        while len(o._items) < len(gt_fns):
            toc = time.time()
            finished = len(o._items)
            if finished > 0:
                print('{}/{} ETA: {}s  Elapsed: {}s'.format(finished, len(gt_fns), (len(gt_fns) - finished) / finished * (toc - tic), toc - tic), end='\r')
            time.sleep(10)
        pool.close()
        pool.join()
        evals = [i[1] for i in o._items]
        hist = np.sum(evals, axis=0)
        ious = per_class_iu(hist).tolist()
        classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'TOTAL']
        ious.append(np.nanmean(ious))
        print('[SEMANTIC]')
        [print(a, '\t\t', b) for a, b in zip(classes, ious)]
        print(','.join([str(i) for i in ious]))

    # drivable area
    if 'drivable' in tasks:
        print('Evaluating drivable...')
        drivable_base = os.path.join(args.data_dir, 'labels', 'drivable', 'drivable_val')
        gt_fns = sorted([os.path.join(drivable_base, d) for d in os.listdir(drivable_base)])

        pred_drivable_base = os.path.join(args.pred_dir, 'drivable')
        drivable_fns = [os.path.join(pred_drivable_base, '{}.npy'.format(n.split('.')[0].split('/')[-1])) for n in gt_fns]

        pool = Pool(10)
        print(len(gt_fns), len(drivable_fns))
        o = pool.imap_unordered(_eval_drivable, zip(gt_fns, drivable_fns))
        tic = time.time()
        while len(o._items) < len(gt_fns):
            toc = time.time()
            finished = len(o._items)
            if finished > 0:
                print('{}/{} ETA: {}s  Elapsed: {}s'.format(finished, len(gt_fns), (len(gt_fns) - finished) / finished * (toc - tic), toc - tic), end='\r')
            time.sleep(10)
        pool.close()
        pool.join()

        drivable_evals = [i[1] for i in o._items]
        # if len(drivable_evals[0][0]) == 9:
        #     lane_evals = np.mean([i[0] for i in drivable_evals], axis=0)
        #     print('[LANE]\n[thresh=10] {} {} {}\n[thresh=5] {} {} {}\n[thresh=1] {} {} {}'.format(*lane_evals))
        #     for e in lane_evals:
        #         print(e)
        drivable_hist = np.sum(drivable_evals, axis=0)
        drivable_ious = per_class_iu(drivable_hist[0]).tolist()
        drivable_ious.append(sum(drivable_ious[1:])/2)
        print('[DRIVABLE]\n[direct] {} [alt] {} [overall] {}'.format(*drivable_ious[1:]))
        for d in drivable_ious:
            print(d)

if __name__ == '__main__':
    main()
