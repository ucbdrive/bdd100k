# set up demo
import os
import subprocess
import json
import numpy as np
import mmcv
import torch
from test_eval_video import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from utils import thin_edge
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import pycocotools.mask as pmask
import time
from skimage import measure
from skimage.morphology import binary_dilation, disk
import argparse

DRIVABLE_AREA_COLORS = np.array([
    [0, 0, 0, 0],
    [217, 83, 79, 150],
    [91, 192, 222, 150]
])

LANE_MARKING_COLORS = np.array([
    [0, 0, 0, 0],
    [255, 0, 0, 255],
    [0, 0, 255, 255]
])

SEMANTIC_COLORS = [
    (128, 64,128),
    (244, 35,232),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]
SEMANTIC_COLORS = np.array([i + (200,) for i in SEMANTIC_COLORS])

DETECTION_COLORS = [
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,255),
    (  0,  0, 180),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (250,170, 30),
    (220,220,  0),
    (119, 11, 32)
]
DETECTION_COLORS = np.array([i + (200,) for i in DETECTION_COLORS]) / 255.

np.random.seed(1)
COLORS = np.array([np.append(np.random.randint(256, size=3)/255., 0.8) for _ in range(1000)])

def get_fig_ax(fig_w=16, fig_h=9, dpi=80):
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    return fig, ax

def dispatch_unordered_tasks(fn, args, pool_size=10):
    pool = Pool(pool_size)
    n = len(args[0])
    out = pool.imap_unordered(fn, zip(*args))
    tic = time.time()
    while len(out._items) < n:
        toc = time.time()
        finished = len(out._items)
        if finished > 0:
            print('{}/{} ETA: {}s  Elapsed: {}s'.format(finished, n, (n - finished) / finished * (toc - tic), toc - tic), end='\r')
        time.sleep(1)
    pool.close()
    pool.join()
    return [o[0] for o in out._items]

def draw_overlaid_label_map(ax, label, ignore_index=255, color_map=None, alpha=0.8):
    if ignore_index > 0:
        label[label == ignore_index] = -1
        label += 1
    if color_map is None:
        color_map = COLORS
    ax.imshow(color_map[label])
    return ax

def draw_bboxs(ax, boxes, labels, colors=None, scale=1, score_threshold=0.5):
    colors = ['red'] * len(boxes) if colors is None else colors
    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2, score = box
        if score < score_threshold:
            continue
        patch = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3 * scale, edgecolor=color, facecolor='none',
            fill=False, alpha=0.75
        )
        ax.add_patch(patch)
    return ax

def draw_polys(ax, polys, scores, labels, facecolors=None, edgecolors=None, scale=1, closed=True, alpha=0.6, score_threshold=0.5):
    facecolors = ['red'] * len(polys) if facecolors is None else facecolors
    edgecolors = ['white'] * len(polys) if edgecolors is None else edgecolors
    polys = np.array(polys)[np.array(scores) > score_threshold]

    for polys, label, facecolor, edgecolor in zip(polys, labels, facecolors, edgecolors):
        for poly in polys:
            points = poly
            codes = [Path.LINETO for v in poly]
            codes[0] = Path.MOVETO

            if closed:
                codes[-1] = Path.CLOSEPOLY

            patch = mpatches.PathPatch(
                Path(points, codes),
                facecolor=facecolor if closed else 'none',
                edgecolor=edgecolor,  # if not closed else 'none',
                lw=1 * scale if closed else 3 * scale, alpha=alpha,
                antialiased=False, snap=True)

            ax.add_patch(patch)

    return ax

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=2):
    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    polygons = [ [ [arr[2*i], arr[2*i+1]] for i in range(len(arr) // 2)] for arr in polygons]
    return polygons

def _frame_based_helper(args):
    det_result, frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # drivable area
    driv_result = np.load(os.path.join(pred_dir, 'driv', '{}.npy'.format(img_name)))
    ax = draw_overlaid_label_map(ax, driv_result[-1], ignore_index=-1, color_map=DRIVABLE_AREA_COLORS)

    # detection
    boxes = [b for r in det_result for b in r]
    labels = [[i] * len(det_result[i]) for i in range(len(det_result))]
    labels = [l for label in labels for l in label]
    # color by label
    colors = DETECTION_COLORS[labels]
    ax = draw_bboxs(ax, boxes, labels, colors)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

def _lane_helper(args):
    frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # lane marking
    lane_result = np.load(os.path.join(pred_dir, 'lane', '{}.npy'.format(img_name)))
    # edge thinning
    # lane_after_process = np.zeros_like(lane_result[0])
    # for l in [1, 2]:
    #     thinned = thin_edge(lane_result[0] == l)
    #     thickened = binary_dilation(thinned, disk(2)) * l
    #     lane_after_process[lane_after_process == 0] = thickened[lane_after_process == 0]
    # removing of attributes?
    lane_after_process = lane_result[0]
    lane_after_process[lane_after_process > 0] = 1
    ax = draw_overlaid_label_map(ax, lane_after_process, ignore_index=-1, color_map=LANE_MARKING_COLORS, alpha=1.0)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

def _ins_seg_helper(args):
    det_result, segm_result, frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # segmentation
    masks_by_class = [ [pmask.decode(mask) for mask in class_segm] for class_segm in segm_result ]
    polys = [ binary_mask_to_polygon(mask) for class_segm in masks_by_class for mask in class_segm ]
    labels = [[i] * len(det_result[i]) for i in range(len(det_result))]
    labels = [l for label in labels for l in label]
    scores = [det['bbox'][-1] for dets in det_result for det in dets]
    # color by label
    colors = DETECTION_COLORS[labels]
    ax = draw_polys(ax, polys, scores, labels, facecolors=colors)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

def _semantic_helper(args):
    frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # semantic segmentation
    sem_seg_result = np.load(os.path.join(pred_dir, 'sem_seg', '{}.npy'.format(img_name)))
    ax = draw_overlaid_label_map(ax, sem_seg_result[-1], ignore_index=-1, color_map=SEMANTIC_COLORS, alpha=0.8)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

def _mot_helper(args):
    track_result, frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # detection
    boxes = [b['bbox'] for b in track_result.values()]
    labels = [b['label'] for b in track_result.values()]
    ids = track_result.keys()
    # color by id
    colors = COLORS[list(ids)]
    ax = draw_bboxs(ax, boxes, ids, colors)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

def _mots_helper(args):
    track_result, segm_track_result, frame_path, pred_dir, vis_out_dir = args
    img_name = frame_path.split('/')[-1].split('.')[0]
    fig, ax = get_fig_ax()
    img = Image.open(frame_path)
    ax.imshow(img)

    # detection
    masks = [ pmask.decode(mask['segmentation']) for mask in segm_track_result.values() ]
    polys = [ binary_mask_to_polygon(mask) for mask in masks ]
    scores = [ mask['bbox'][-1] for mask in segm_track_result.values() ]
    labels = [b['label'] for b in track_result.values()]
    ids = track_result.keys()
    # color by id
    colors = COLORS[list(ids)]
    ax = draw_polys(ax, polys, scores, labels, facecolors=colors)
    fig.savefig(os.path.join(vis_out_dir, img_name+'.jpg'))
    plt.close()

class InferenceDemo:

    def __init__(self, args, scale=1, fig_w=16, fig_h=9, dpi=80):

        for p in [args.work_dir, args.frame_dir, args.tmp_dir, args.vis_dir]:
            os.makedirs(p, exist_ok=True)

        self.cfg = mmcv.Config.fromfile(args.cfg_fn)
        self.work_dir = args.work_dir
        self.frame_dir = args.frame_dir
        self.frame_paths = sorted(os.listdir(self.frame_dir))
        self.tmp_dir = args.tmp_dir
        self.pred_dir = args.pred_dir
        self.vis_dir = args.vis_dir
        self.lane_ckpt = args.lane_ckpt
        self.driv_ckpt = args.driv_ckpt
        self.sem_seg_ckpt = args.sem_seg_ckpt
        self.det_ckpt = args.det_ckpt
        self.ins_seg_ckpt = args.ins_seg_ckpt
        self.box_track_ckpt = args.box_track_ckpt
        self.seg_track_ckpt = args.seg_track_ckpt
        self.video_name = '-'.join(os.listdir(self.frame_dir)[0].split('-')[:-1])
        self.scale = scale
        self.fig_w = fig_w
        self.fig_h = fig_h
        self.dpi = dpi

    def get_inference_results(self):
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        self.cfg.model.pretrained = None

        # data
        categories = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motorcycle', 'traffic light', 'traffic sign', 'train']
        tmp_ann = dict(
            categories=[dict(id=i+1, name=c) for i, c in enumerate(categories)],
            images=[{'file_name': f, 'height': 720, 'width': 1280, 'id': i, 'video_id': 0, 'index': i} for i, f in enumerate(self.frame_paths)],
            videos=[dict(id=0, name=self.video_name)]
        )
        self.out_json_fn = os.path.join(self.tmp_dir, 'tmp_list.json')
        with open(self.out_json_fn, 'w') as f:
            json.dump(tmp_ann, f)

        self.cfg.out_path = self.pred_dir

        print('Building model...')
        # build the model and load checkpoint
        self.model = build_detector(
            self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg, global_cfg=self.cfg)
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)

        # build the dataloader
        self.model = MMDataParallel(self.model.cuda(), device_ids=[0])

        # get results for each task
        self.get_drivable_results()
        self.get_lane_results()
        self.get_sem_seg_results()
        self.get_det_results()
        self.get_ins_seg_results()
        self.get_box_track_results()
        self.get_seg_track_results()

    def get_lane_results(self):
        # lane marking
        print('Inferencing lane marking...')
        dataset_cfg = dict(type='BddStreet',
                            image_dir=self.frame_dir,
                            label_dir='',
                            phase='val',
                            flip_ratio=0,
                            with_lane=True,
                            with_drivable=False,
                            img_prefix=self.frame_dir,
                            img_scale=(1280, 720),
                            img_norm_cfg=self.cfg.img_norm_cfg,
                            size_divisor=32,
                            with_label=False,
                            test_mode=True,
                            ann_file=None,
                            task='lane'
                        )
        dataset = build_dataset(dataset_cfg)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)
        print(self.lane_ckpt)
        checkpoint = load_checkpoint(self.model, self.lane_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader, os.path.join(self.pred_dir, 'lane'))

    def get_drivable_results(self):
        # drivable area
        print('Inferencing drivable area...')
        dataset_cfg = dict(type='BddStreet',
                            image_dir=self.frame_dir,
                            label_dir='',
                            phase='val',
                            flip_ratio=0,
                            with_lane=False,
                            with_drivable=True,
                            img_prefix=self.frame_dir,
                            img_scale=(1280, 720),
                            img_norm_cfg=self.cfg.img_norm_cfg,
                            size_divisor=32,
                            with_label=False,
                            test_mode=True,
                            ann_file=None,
                            task='drivable'
                        )

        dataset = build_dataset(dataset_cfg)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.driv_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader, os.path.join(self.pred_dir, 'driv'))

    def get_sem_seg_results(self):
        # semantic segmentation
        print('Inferencing semantic segmentation...')
        dataset_cfg = dict(type='BddSemanticSeg',
                            image_dir=self.frame_dir,
                            label_dir='',
                            phase='val',
                            flip_ratio=0,
                            img_prefix=self.frame_dir,
                            img_scale=(1280, 720),
                            img_norm_cfg=self.cfg.img_norm_cfg,
                            size_divisor=32,
                            with_label=False,
                            test_mode=True,
                            ann_file=None,
                            task='sem_seg'
                        )

        dataset = build_dataset(dataset_cfg)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.sem_seg_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader, self.pred_dir)

    def get_det_results(self):
        # detection
        print('Inferencing detection...')
        dataset_cfg = dict(
                        type='CocoDataset',
                        ann_file=self.out_json_fn,
                        img_prefix=self.frame_dir,
                        img_scale=(1280, 720),
                        img_norm_cfg=self.cfg.img_norm_cfg,
                        size_divisor=32,
                        flip_ratio=0,
                        with_mask=False,
                        with_label=False,
                        test_mode=True,
                        task='det'
                    )

        dataset = build_dataset(dataset_cfg)
        self.model.CLASSES = dataset.CLASSES
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.det_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader)
        mmcv.dump(results, os.path.join(self.pred_dir, 'det.pkl'))

    def get_ins_seg_results(self):
        # instance segmentation
        print('Inferencing instance segmentation...')
        dataset_cfg = dict(
                        type='CocoDataset',
                        ann_file=self.out_json_fn,
                        img_prefix=self.frame_dir,
                        img_scale=(1280, 720),
                        img_norm_cfg=self.cfg.img_norm_cfg,
                        size_divisor=32,
                        flip_ratio=0,
                        with_mask=True,
                        with_label=False,
                        test_mode=True,
                        task='ins_seg'
                    )

        dataset = build_dataset(dataset_cfg)
        self.model.CLASSES = dataset.CLASSES
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.ins_seg_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader)
        mmcv.dump(results, os.path.join(self.pred_dir, 'ins_seg.pkl'))

    def get_box_track_results(self):
        # multiple object tracking
        print('Inferencing multiple object tracking...')
        dataset_cfg = dict(
                        type='BDDVideo',
                        ann_file=self.out_json_fn,
                        img_prefix=self.frame_dir,
                        img_scale=(1280, 720),
                        img_norm_cfg=self.cfg.img_norm_cfg,
                        size_divisor=32,
                        flip_ratio=0,
                        with_mask=False,
                        with_label=False,
                        with_track=True,
                        test_mode=True,
                        task='box_track'
                    )

        dataset = build_dataset(dataset_cfg)
        self.model.CLASSES = dataset.CLASSES
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.box_track_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader)
        mmcv.dump(results, os.path.join(self.pred_dir, 'box_track.pkl'))

    def get_seg_track_results(self):
        # multiple object tracking with segmentation
        print('Inferencing multiple object tracking with segmentation...')
        dataset_cfg = dict(
                        type='BDDVideo',
                        ann_file=self.out_json_fn,
                        img_prefix=self.frame_dir,
                        img_scale=(1280, 720),
                        img_norm_cfg=self.cfg.img_norm_cfg,
                        size_divisor=32,
                        flip_ratio=0,
                        with_mask=True,
                        with_label=False,
                        with_track=True,
                        test_mode=True,
                        task='seg_track'
                    )

        dataset = build_dataset(dataset_cfg)
        self.model.CLASSES = dataset.CLASSES
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        checkpoint = load_checkpoint(self.model, self.seg_track_ckpt, map_location='cpu');
        results = single_gpu_test(self.model, data_loader)
        mmcv.dump(results, os.path.join(self.pred_dir, 'seg_track.pkl'))

    def get_visualizations(self):
        # self.get_raw_video()
        self.get_frame_based_video()
        self.get_ins_seg_video()
        self.get_semantic_video()
        self.get_lane_video()
        self.get_mot_video()
        self.get_mots_video()
        plt.close()

    def get_raw_video(self):
        # get the raw video
        print('Generating the raw video...')
        out_fn = os.path.join(self.vis_dir, 'raw.mp4')
        cmd = 'ffmpeg -r 5 -i {}-%07d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p {}'.format(
            os.path.join(self.frame_dir, self.video_name), out_fn)
        subprocess.call(cmd, shell=True)

    def get_frame_based_video(self):
        # drivable area and detection
        det_results = mmcv.load(os.path.join(self.pred_dir, 'det.pkl'))
        vis_out_dir = os.path.join(self.vis_dir, 'frame_based')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the frame-based result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_frame_based_helper, [det_results['bbox_results'],
                                 [os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])

    def get_lane_video(self):
        # drivable area and detection
        vis_out_dir = os.path.join(self.vis_dir, 'lane')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the lane marking result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_lane_helper,
                                 [[os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])

    def get_ins_seg_video(self):
        # drivable area and detection
        ins_seg_results = mmcv.load(os.path.join(self.pred_dir, 'ins_seg.pkl'))
        vis_out_dir = os.path.join(self.vis_dir, 'ins_seg')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the instance segmentation result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_ins_seg_helper, [ins_seg_results['bbox_results'], ins_seg_results['segm_results'],
                                 [os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])

    def get_semantic_video(self):
        # get the video with semantic segmentation at each frame
        vis_out_dir = os.path.join(self.vis_dir, 'semantic')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the semantic result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_semantic_helper, [[os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])

    def get_mot_video(self):
        # get the multiple object tracking result video
        mot_results = mmcv.load(os.path.join(self.pred_dir, 'box_track.pkl'))
        vis_out_dir = os.path.join(self.vis_dir, 'mot')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the MOT result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_mot_helper, [mot_results['track_results'],
                                 [os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])


    def get_mots_video(self):
        # get the multiple object tracking with segmentation result video
        mots_results = mmcv.load(os.path.join(self.pred_dir, 'seg_track.pkl'))
        vis_out_dir = os.path.join(self.vis_dir, 'mots')
        os.makedirs(vis_out_dir, exist_ok=True)
        print('Generating the MOTS result video...')
        n = len(self.frame_paths)
        dispatch_unordered_tasks(_mots_helper, [mots_results['track_results'], mots_results['segm_track_results'],
                                 [os.path.join(self.frame_dir, i) for i in self.frame_paths], [self.pred_dir] * n, [vis_out_dir] * n])

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg-fn', '-c', help='the config file name', default='configs/mtl/bdd_dla34up_lrsditx.py')
    parser.add_argument('--work-dir', '-w', help='the work directory', default='./work_dirs/bdd_demo')
    parser.add_argument('--frame-dir', '-f', help='the directory of input frames', default='/shared/haofeng/bdd-tracking-2k/images/train/000f8d37-d4c09a0f')# './work_dirs/bdd_demo/example_video')
    parser.add_argument('--tmp-dir', '-tmp', help='the temporary directory', default='/tmp')
    parser.add_argument('--pred-dir', '-p', help='the directory to predictions', default='./work_dirs/predictions')
    parser.add_argument('--vis-dir', '-v', help='the directory to visualizations', default='./work_dirs/visualizations')
    parser.add_argument('--lane-ckpt', '-l', help='the checkpoint of the lane marking model', default='weights/lr_dla34.pth')
    parser.add_argument('--driv-ckpt', '-r', help='the checkpoint of the drivable area model', default='weights/lr_dla34.pth')
    parser.add_argument('--sem-seg-ckpt', '-s', help='the checkpoint of the semantic segmentation model', default='weights/s_dla34.pth')
    parser.add_argument('--det-ckpt', '-d', help='the checkpoint of the detection model', default='weights/d_dla34.pth')
    parser.add_argument('--ins-seg-ckpt', '-i', help='the checkpoint of the instance segmentation model', default='weights/i_dla34.pth')
    parser.add_argument('--box-track-ckpt', '-t', help='the checkpoint of the MOT model', default='/shared/haofeng/github/modat/work_dirs/debug/BDD-t-2x/dla34up/epoch_12.pth')
    parser.add_argument('--seg-track-ckpt', '-x', help='the checkpoint of the MOTS model', default='/shared/haofeng/github/modat/work_dirs/debug/BDD-ditx.pth')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    demo = InferenceDemo(args)

    # run inference
    tic = time.time()
    # demo.get_inference_results()
    toc = time.time()
    inference_time = toc - tic
    # get visualizations
    demo.get_visualizations()
    tic = time.time()
    visualize_time = tic - toc
    print('\n\n\nTotal inference time: {}s\nTotal visualize time: {}s'.format(inference_time, visualize_time))
