import argparse
import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from collections import defaultdict
from mmdet.core import finding_video_tubes, draw_results, mdat_eval
import numpy as np
import torch.nn.functional as F
from PIL import Image
import pandas as pd


def single_gpu_test(model, data_loader, out_dir=None, show=False):
    model.eval()
    # if os.isdir(out_dir):
    #     os.makedirs(out_dir, exist_ok=True)
    outputs = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    cls_conf_mats = None
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        # dump results for segmentation tasks
        if type(result) == dict and len(np.intersect1d(list(result.keys()), ['lane_results', 'drivable_results', 'sem_seg_results'])) > 0:
            img_name = data['img_meta'][0].data[0][0]['file_name'].split('/')[-1].split('.')[0]
            # remove padding for segmentation
            for k in result.keys():
                if k == 'lane_results':
                    result[k] = [F.softmax(r[:, :, :720, :], dim=1) for r in result[k]]
                else:
                    result[k] = result[k][:, :, :720, :]
            # sem_seg results
            if 'sem_seg_results' in result.keys() and out_dir:
                os.makedirs(os.path.join(out_dir, 'sem_seg'), exist_ok=True)
                np.save(os.path.join(out_dir, 'sem_seg', '{}.npy'.format(img_name)), result['sem_seg_results'].squeeze(1).astype(np.uint8))
            # lane and drivable area
            if 'lane_results' in result.keys():
                # to reduce output size, quantize to uint8
                lane_results = [(255 * r).squeeze(0)[1:].cpu().data.numpy().astype(np.uint8) for r in result['lane_results']]
                # direction: 2 channels, continuity: 2 channels, category: 8 channels
                # p1: direction and continuity
                p1 = Image.fromarray(np.concatenate(lane_results[:2]).transpose(1,2,0), mode='RGBA')
                # p2 and p3: category
                p2 = Image.fromarray(lane_results[2][:4].transpose(1,2,0), mode='RGBA')
                p3 = Image.fromarray(lane_results[2][4:].transpose(1,2,0), mode='RGBA')
                os.makedirs(os.path.join(out_dir, 'lane'), exist_ok=True)
                p1.save(os.path.join(out_dir, 'lane', '{}_1.png'.format(img_name)))
                p2.save(os.path.join(out_dir, 'lane', '{}_2.png'.format(img_name)))
                p3.save(os.path.join(out_dir, 'lane', '{}_3.png'.format(img_name)))
            if 'drivable_results' in result.keys():
                drivable_results = result['drivable_results'].squeeze(1)
                os.makedirs(os.path.join(out_dir, 'drivable'), exist_ok=True)
                np.save(os.path.join(out_dir, 'drivable', '{}.npy'.format(img_name)), np.array(drivable_results))
            for _ in range(data['img'][0].size(0)):
                prog_bar.update()
            continue
        elif 'cls_results' in result.keys():
            if cls_conf_mats is None:
                cls_conf_mats = [np.zeros((6, 6)), np.zeros((6, 6)), np.zeros((3, 3))]
            gt_cls = data['img_meta'][0].data[0][0]['gt_cls'].data.numpy()
            for i in range(3):
                if gt_cls[i] > 0:
                    cls_conf_mats[i][gt_cls[i] - 1][result['cls_results'][i] - 1] += 1
        # for instance-based tasks, gather outputs
        for k in result.keys():
            outputs[k].append(result[k])
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    if cls_conf_mats:
        attrs = dict(
            weather=['rainy', 'snowy', 'clear', 'overcast', 'partly cloudy', 'foggy'],
            scene=['tunnel', 'residential', 'parking lot', 'city street', 'gas station', 'highway'],
            timeofday=['daytime', 'night', 'dawn/dusk']
        )
        for a, m in zip(attrs, cls_conf_mats):
            accs = m.diagonal() / np.sum(m, axis=1)
            macc = np.mean(accs)
            df = pd.DataFrame(data={k: [v] for k, v in zip(attrs[a] + ['MEAN'], accs.tolist() + [macc])})
            print('\n', a)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)
        return

    return outputs


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--det_ckpt', type=str, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    for t in cfg.data.test:
        t.test_mode = True
    cfg.out_path = args.out.split('.pkl')[0] if args.out else None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg, global_cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.det_ckpt is not None:
        print('Loading detection models...')
        det_ckpt = load_checkpoint(model, args.det_ckpt, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if not type(cfg.data.test) == list:
        cfg.data.test = [cfg.data.test]

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(model.cuda())

    outputs = dict()
    for c in cfg.data.test:
        dataset = build_dataset(c)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        if not distributed:
            results = single_gpu_test(model, data_loader, args.out, args.show)
            if results is not None:
                outputs.update(results)
        else:
            outputs.update(multi_gpu_test(model, data_loader, args.tmpdir))

    rank, _ = get_dist_info()
    if len(outputs.keys()) > 0 and args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        if not (args.out.endswith('.pkl') or args.out.endswith('.json')):
            args.out += '.pkl'
        if 'track_results' in outputs.keys():
            mmcv.dump(outputs['track_results'], args.out)
        else:
            mmcv.dump(outputs, args.out)
        if 'bbox_results' in outputs.keys():
            result_files = results2json(dataset, outputs['bbox_results'], args.out)
            coco_eval(result_files, ['bbox', 'segm'], cfg.data.test[0].ann_file)
        if 'segm_results' in outputs.keys():
            result_files = results2json(dataset, [(b, s) for b, s in zip(outputs['bbox_results'], outputs['segm_results'])], args.out)
            coco_eval(result_files, ['segm'], cfg.data.test[0].ann_file)
        # if 'new_bbox_results' in outputs.keys():
        #     # For tracking
        #     result_files = results2json(dataset, outputs['new_bbox_results'],
        #                                 args.out)
        #     coco_eval(result_files, ['bbox'], cfg.data.test[0].ann_file)
        if 'track_results' in outputs.keys():
            print("Evaluating box tracking...")
            mdat_eval(outputs['track_results'], dataset, args.out, cfg)
        if 'segm_track_results' in outputs.keys():
            print("Evaluating segmentation tracking...")
            mdat_eval(outputs['segm_track_results'], dataset, args.out, cfg, with_mask=True)


if __name__ == '__main__':
    main()
