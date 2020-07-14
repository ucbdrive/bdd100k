import os
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, coco_eval, fast_eval_recall
from .mean_ap import eval_map
from mmdet import datasets
from ..track import mdat_eval


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()
        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class BDDEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.val_cfg = dataset
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        if runner.rank == 0:
            outputs = defaultdict(list)
            prog_bar = mmcv.ProgressBar(len(self.dataset))
            for idx in range(len(self.dataset)):
                data = self.dataset[idx]
                data_gpu = scatter(
                    collate([data], samples_per_gpu=1),
                    [torch.cuda.current_device()])[0]
                # compute output
                with torch.no_grad():
                    result = runner.model(
                        return_loss=False, rescale=True, **data_gpu)
                outputs['bbox_results'].append(result['bbox_results'])
                outputs['new_bbox_results'].append(result['new_bbox_results'])
                outputs['track_results'].append(result['track_results'])
                if 'segm_results' in result.keys():
                    outputs['bbox_results'][-1] = (outputs['bbox_results'][-1], result['segm_results'])
                    outputs['segm_track_results'].append(result['segm_track_results'])
                prog_bar.update()
            out_name = '{}/tmp.pkl'.format(runner.work_dir)
            print('\nwriting results to {}'.format(out_name))
            mmcv.dump(outputs, out_name)

            # bbox
            result_files = results2json(self.dataset, outputs['bbox_results'],
                                        out_name)
            coco_eval(result_files, ['bbox'], self.val_cfg.ann_file)
            # Box tracking
            result_files = results2json(self.dataset,
                                        outputs['new_bbox_results'], out_name)
            coco_eval(result_files, ['bbox'], self.val_cfg.ann_file)
            print("Evaluating box tracking...")
            mdat_eval_out = mdat_eval(outputs['track_results'], self.dataset, out_name, ann_file=self.val_cfg.ann_file)
            if 'segm_track_results' in outputs.keys():
                print("Evaluating seg tracking...")
                out = mdat_eval(outputs['segm_track_results'], self.dataset, out_name, ann_file=self.val_cfg.ann_file, with_mask=True)
                out = {'seg_' + k: v for k, v in out.items()}
                mdat_eval_out.update(out)
            runner.log_buffer.output.update(mdat_eval_out)
            runner.log_buffer.ready = True


class BddSegEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.val_cfg = dataset
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.task = self.val_cfg.type

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        if self.task == 'BddStreet':
            lane_bins = [np.zeros((n, n)) for n in [3, 3, 9]]
            drivable_bin = np.zeros((3, 3))
        elif self.task == 'BddSemanticSeg':
            sem_seg_bin = np.zeros((20, 20))
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
            for idx in range(len(self.dataset)):
                data = self.dataset[idx]
                data_gpu = scatter(
                    collate([data], samples_per_gpu=1),
                    [torch.cuda.current_device()])[0]
                # compute output
                with torch.no_grad():
                    result = runner.model(
                        return_loss=False, rescale=True, **data_gpu)
                # evaluation
                if self.task == 'BddStreet':
                    if self.dataset.with_lane:
                        _lane_bins = self.eval_lane(result['lane_results'], data['img_meta'][0].data['gt_lane'])
                        for i, bin in enumerate(_lane_bins):
                            lane_bins[i] += bin
                    if self.dataset.with_drivable:
                        drivable_bin += self.eval_drivable(result['drivable_results'], data['img_meta'][0].data['gt_drivable'])
                elif self.task == 'BddSemanticSeg':
                    sem_seg_bin += self.eval_sem_seg(result['sem_seg_results'], data['img_meta'][0].data['gt_sem_seg'])
                prog_bar.update()
            # lane_IoUs = [self.fg_IoUs(b) for b in lane_bins]
            # drivable_IoUs = self.fg_IoUs(drivable_bin)
            outputs = dict()
            if self.task == 'BddStreet':
                print('\nSTREET EVALUATION')
                if self.dataset.with_lane:
                    lane_mIoUs = [self.fg_mIoU(bin) for bin in lane_bins]
                    print('[lane] direction: {} style: {} type: {}'.format(*lane_mIoUs))
                    runner.log_buffer.output.update(dict(val_lane_0=lane_mIoUs[0], val_lane_1=lane_mIoUs[1], val_lane_2=lane_mIoUs[2]))
                    lane_avg_recall = [self.fg_avg_recall(bin) for bin in lane_bins]
                    print('[lane] direction: {} style: {} type: {}'.format(*lane_avg_recall))
                    runner.log_buffer.output.update(dict(val_lane_ar_0=lane_avg_recall[0], val_lane_ar_1=lane_avg_recall[1], val_lane_ar_2=lane_avg_recall[2]))
                if self.dataset.with_drivable:
                    drivable_mIoU = self.fg_mIoU(drivable_bin)
                    print('[driv] mIoU: {}'.format(drivable_mIoU))
                    runner.log_buffer.output.update(dict(val_drivable=drivable_mIoU))
            elif self.task == 'BddSemanticSeg':
                sem_seg_mIoU = self.fg_mIoU(sem_seg_bin)
                print('\nSEMANTIC SEG EVALUATION\n[sem_seg] mIoU: {}'.format(sem_seg_mIoU))
                runner.log_buffer.output.update(dict(val_sem_seg=sem_seg_mIoU))
        runner.log_buffer.ready = True


    def fast_hist(self, pred, label, n):
        pred = pred.flatten()
        label = label.flatten()
        k = (label >= 0) & (label < n)
        return np.bincount(
            n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n) / float(pred.size)

    def fg_recall(self, bin):
        # bin: nxn, 0 being background
        # per_class_rc = np.diag(bin) / float(bin.sum(0))
        # return per_class_rc[1:] * 100
        return [bin[i, i] / sum(bin[i]) * 100 for i in range(1, len(bin))]

    def fg_IoUs(self, bin):
        # bin: nxn, 0 being background
        # per_class_iu = np.diag(bin) / float(bin.sum(0) + bin.sum(1) - np.diag(bin))
        # return per_class_iu[1:] * 100
        return [bin[i, i] / (sum(bin[i]) + sum(bin[:, i]) - bin[i, i]) * 100 for i in range(1, len(bin))]

    def fg_mIoU(self, bin):
        print(self.fg_IoUs(bin))
        return np.nanmean(self.fg_IoUs(bin))

    def fg_avg_recall(self, bin):
        return np.nanmean(self.fg_recall(bin))

    def eval_lane(self, pred, target):
        # Lane direction, style, type
        assert len(pred) == 3
        return [self.fast_hist(pred[i], target[:, :, i], n) for i, n in enumerate([3, 3, 9])]

    def eval_drivable(self, pred, target):
        return self.fast_hist(pred, target, 3)

    def eval_sem_seg(self, pred, target):
        return self.fast_hist(pred, target, 20)


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'
                     ] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            if not res_type in result_files.keys(): continue
            cocoDt = cocoGt.loadRes(result_files[res_type])
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            if not res_type in result_files.keys(): continue
            os.remove(result_files[res_type])
