import os.path as osp
import random
import mmcv
import numpy as np
from .custom import CustomDataset
from mmcv.parallel import DataContainer as DC
from pycocotools.bdd import BDD
from .utils import to_tensor, random_scale, random_crop
from .registry import DATASETS
from .builder import build_dataset
import os
from PIL import Image
import json
import torch


@DATASETS.register_module
class BddStreet(CustomDataset):

    def __init__(self,
                 image_dir,
                 label_dir,
                 phase='train',
                 flip_ratio=0.5,
                 with_lane=True,
                 with_drivable=True,
                 test_mode=False,
                 crop_size=None,
                 task='street',
                 *args,
                 **kwargs):
        super(BddStreet, self).__init__(*args, **kwargs)
        # data
        self.test_mode = test_mode
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase.split('_')[-1]
        self.crop_size = crop_size
        self.img_infos = self.get_img_infos()
        if len(phase.split('_')) > 1:
            if (phase.startswith('mini_')):
                self.img_infos = self.img_infos[:1000]
            else:
                num = int(phase.split('_')[0])
                self.img_infos = self.img_infos[:num]
        self.flip_ratio = flip_ratio
        self.with_lane = with_lane
        self.with_drivable = with_drivable
        self.flag = np.ones(len(self), dtype=np.uint8)
        self.task = task


    def __len__(self):
        return len(self.img_infos)

    def get_img_infos(self):
        image_list = [i for i in os.listdir(self.image_dir) if i.endswith('.jpg')]
        img_infos = []
        for i, fn in enumerate(image_list):
            print(self.label_dir, fn)
            img_infos.append(dict(
                file_name = os.path.join(self.image_dir, fn),
                street_file_name = os.path.join(self.label_dir, fn[:-3] + 'png'),
                height = 720,
                width = 1280
            ))

        return img_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)

        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(img_info['file_name'])

        # load labels
        if os.path.exists(img_info['street_file_name']):
            lbl = np.array(Image.open(img_info['street_file_name']), dtype=np.uint8)
            if self.with_lane:
                gt_lane = lbl[:, :, :-1]
            if self.with_drivable:
                gt_drivable = lbl[:, :, -1:]
        else:
            h, w = img.shape[:2]
            if self.with_lane:
                gt_lane = np.zeros((h, w, 3), dtype=np.uint8)
            if self.with_drivable:
                gt_drivable = np.zeros((h, w, 1), dtype=np.uint8)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        crop_info = random_crop([img_info['width'], img_info['height']], self.crop_size) if self.crop_size else None
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, crop_info=crop_info, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        if self.with_lane:
            gt_lane = self.seg_transform(gt_lane, img_scale,
                                         flip, crop_info=crop_info, pad_val=255)
            gt_lane = gt_lane.copy()
        if self.with_drivable:
            gt_drivable = self.seg_transform(gt_drivable, img_scale,
                                             flip, crop_info=crop_info, pad_val=255)
            gt_drivable = gt_drivable.copy()
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True))
        if self.with_lane:
            data['gt_lane'] = DC(to_tensor(gt_lane))
        if self.with_drivable:
            data['gt_drivable'] = DC(to_tensor(gt_drivable))

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(img_info['file_name'])

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)

            # load labels
            # if not self.test_mode:
            if os.path.exists(img_info['street_file_name']):
                _lbl = mmcv.imread(img_info['street_file_name'])
            else:
                _lbl = np.zeros_like(_img).transpose(1,2,0)
            _img = to_tensor(_img)
            _lbl = self.seg_transform(_lbl, scale, flip,  pad_val=255)

            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                task=self.task,
                gt_lane=_lbl[:, :, :-1],
                gt_drivable=_lbl[:, :, -1:],
                file_name=img_info['file_name'])
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(
                img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(
                    img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data


@DATASETS.register_module
class BddSemanticSeg(CustomDataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 phase='train',
                 flip_ratio=0.5,
                 test_mode=False,
                 crop_size=None,
                 *args,
                 **kwargs):
        super(BddSemanticSeg, self).__init__(*args, **kwargs)
        # data
        self.test_mode = test_mode
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase.split('_')[-1]
        self.img_infos = self.get_img_infos()
        if len(phase.split('_')) > 1:
            if (phase.startswith('mini_')):
                self.img_infos = self.img_infos[:1000]
            else:
                num = int(phase.split('_')[0])
                self.img_infos = self.img_infos[:num]
        self.flip_ratio = flip_ratio
        self.crop_size = crop_size
        self.flag = np.ones(len(self), dtype=np.uint8)


    def __len__(self):
        return len(self.img_infos)

    def get_img_infos(self):
        image_list = [i for i in os.listdir(self.image_dir) if i.endswith('.jpg')]
        if self.phase == 'train':
            image_list = [i for i in image_list if not (i in '/shared/haofeng/github/bdd100k/images/val')]
        elif self.phase == 'val':
            image_list = [i for i in image_list if not (i in '/shared/haofeng/github/bdd100k/images/train')]
        img_infos = []
        for i, fn in enumerate(image_list):
            img_infos.append(dict(
                file_name = os.path.join(self.image_dir, fn),
                label_file_name = os.path.join(self.label_dir, fn[:-4] + '_train_id.png'),
                height = 720,
                width = 1280
            ))

        return img_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)

        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(img_info['file_name'])
        lbl = mmcv.imread(img_info['label_file_name'], flag='grayscale')

        # # add background class
        # lbl[lbl == 255] = 254
        # lbl += 1
        # lbl[lbl == 255] = 0

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        crop_info = random_crop([img_info['width'], img_info['height']], self.crop_size) if self.crop_size else None

        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, crop_info=crop_info, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        gt_sem_seg = self.seg_transform(lbl, img_scale,
                                        flip, crop_info=crop_info, pad_val=255)
        gt_sem_seg = gt_sem_seg.copy()

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_sem_seg=DC(to_tensor(gt_sem_seg)))

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(img_info['file_name'])
        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            # load labels
            # if not self.test_mode:
            if os.path.exists(img_info['label_file_name']):
                _lbl = mmcv.imread(img_info['label_file_name'], flag='grayscale')
                _lbl = self.seg_transform(_lbl, scale, flip, pad_val=255)
            else:
                # import pdb; pdb.set_trace()
                _lbl = torch.zeros_like(_img)

            # # add background class
            # _lbl[_lbl == 255] = 254
            # _lbl += 1
            # _lbl[_lbl == 255] = 0

            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                task='sem_seg',
                gt_sem_seg= _lbl,
                file_name=img_info['file_name'])
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(
                img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(
                    img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data


@DATASETS.register_module
class BddCls(CustomDataset):

    def __init__(self,
                 image_dir,
                 label_dir,
                 phase='train',
                 flip_ratio=0.5,
                 with_lane=True,
                 with_drivable=True,
                 test_mode=False,
                 crop_size=None,
                 task='cls',
                 *args,
                 **kwargs):
        super(BddCls, self).__init__(*args, **kwargs)
        # data
        self.test_mode = test_mode
        self.image_dir = image_dir
        with open(label_dir) as f:
            self.labels = json.load(f)
        # temporary
        for k in self.labels.keys():
            self.labels[k] = self.labels[k]
        self.phase = phase.split('_')[-1]
        self.flip_ratio = flip_ratio
        self.flag = np.ones(len(self), dtype=np.uint8)
        self.task = task


    def __len__(self):
        return len(self.labels['names'])


    def __getitem__(self, idx):
        # load image
        img = mmcv.imread(os.path.join(self.img_prefix, self.labels['names'][idx]))
        ori_shape = img.shape
        flip = np.random.rand() < self.flip_ratio
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, 1, flip)
        # if not self.phase == 'test':
        gt_cls = np.array([
            self.labels['weather'][idx],
            self.labels['scene'][idx],
            self.labels['timeofday'][idx]])
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            task='cls',
            file_name=self.labels['names'][idx],
            gt_cls=DC(to_tensor(gt_cls)))
        data = dict(
            img=DC(to_tensor(img), stack=True) if self.phase != 'test' else [img],
            img_meta=DC(img_meta, cpu_only=True) if self.phase != 'test' else [DC(img_meta, cpu_only=True)])
        if self.phase != 'test':
            data['gt_cls'] = DC(to_tensor(gt_cls))
        return data


@DATASETS.register_module
class BddDetection(CustomDataset):
    pass


@DATASETS.register_module
class BddInstanceSeg(CustomDataset):
    pass


@DATASETS.register_module
class BddBoxTracking(CustomDataset):
    CLASSES = ('person', 'rider', 'car', 'bus', 'truck', 'bike', 'motorcycle')

    def __init__(self,
                 train_sample_interval=None,
                 aug_ref_bbox_param=None,
                 *args,
                 **kwargs):
        super(BddBoxTracking, self).__init__(*args, **kwargs)
        # tracking parameters
        self.train_sample_interval = train_sample_interval
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # load video annotations
        self.vid_infos = self.load_vid_anns()
        if self.test_mode:
            self.serial_img_infos = self.parse_test_infos()

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        self.bdd = BDD(ann_file)

        # Categories
        self.cat_ids = self.bdd.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        # Image Infos
        self.img_ids = self.bdd.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.bdd.loadImgs([i])[0]
            info['file_name'] = info['file_name']
            img_infos.append(info)

        return img_infos

    def parse_test_infos(self):
        img_infos = []
        for vid_id in self.vid_ids:
            img_ids = self.bdd.getImgIdsFromVideoId(vid_id)
            for index, img_id in enumerate(img_ids):
                info = self.bdd.loadImgs([img_id])[0]
                info['first_frame'] = True if index == 0 else False
                img_infos.append(info)
        return img_infos

    def load_vid_anns(self):
        self.vid_ids = self.bdd.getVideoIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.bdd.loadVideos([i])[0]
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, img_id):
        ann_ids = self.bdd.getAnnIds(imgIds=[img_id])
        ann_info = self.bdd.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]

        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2 * np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2 * np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:, :2] + bbox[:, 2:]) / 2.
        sizes = bbox[:, 2:] - bbox[:, :2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes / 2.
        new_x2y2 = new_centers + new_sizes / 2.
        c_min = [0, 0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx, vid_id):
        # sample another frame in the same sequence as reference
        img_info = self.img_infos[idx]
        index = img_info['index']
        img_ids = self.bdd.getImgIdsFromVideoId(vid_id)
        if self.train_sample_interval is None:
            self.train_sample_interval = len(img_ids)
        valid_ids = []
        for k, id in enumerate(img_ids):
            if (abs(k - index) <= self.train_sample_interval) and (k != index):
                valid_ids.append(id)
        return random.choice(valid_ids)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(os.path.join(self.img_prefix, img_info['file_name']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(img_info['id'])
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        # load reference image
        vid_id = img_info['video_id']
        ref_img_id = self.sample_ref(idx, vid_id)
        ref_img_info = self.bdd.loadImgs([ref_img_id])[0]
        ref_img = mmcv.imread(
            os.path.join(self.img_prefix, ref_img_info['file_name']))
        ref_ann = self.get_ann_info(ref_img_id)
        ref_bboxes = ref_ann['bboxes']
        if len(ref_bboxes) == 0:
            return None
        # obj ids attribute does not exist in current annotation
        # need to add it
        ref_instances = ref_ann['instance_ids']
        gt_instances = ann['instance_ids']
        # compute matching of reference frame with current frame
        # -1 denote there is no matching
        gt_pids = [
            ref_instances.index(i) if i in ref_instances else -1
            for i in gt_instances
        ]

        if sum(gt_pids) == -1 * len(gt_pids):
            return None

        if gt_pids:
            gt_pids = np.array(gt_pids, dtype=np.int64)
        else:
            gt_pids = np.array([], dtype=np.int64)

        ref_img, ref_img_shape, _, ref_scale_factor = self.img_transform(
            ref_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        ref_img = ref_img.copy()
        ref_bboxes = self.bbox_transform(ref_bboxes, ref_img_shape,
                                         ref_scale_factor, flip)
        if self.aug_ref_bbox_param is not None:
            ref_bboxes = self.bbox_aug(ref_bboxes, ref_img_shape)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_track:
            data['ref_img'] = DC(to_tensor(ref_img), stack=True)
            data['ref_bboxes'] = DC(to_tensor(ref_bboxes))
            data['gt_pids'] = DC(to_tensor(gt_pids))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.serial_img_infos[idx]
        img = mmcv.imread(os.path.join(self.img_prefix, img_info['file_name']))

        # TODO: Not support props now. Need to re-index if include props
        proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                first_frame=img_info['first_frame'],
                video_id=img_info['video_id'],
                frame_id=img_info['index'],
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_instances = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_instances.append(ann['instance_id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.bdd.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            instance_ids=gt_instances,
            bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann


@DATASETS.register_module
class BddSegTracking(CustomDataset):
    pass
