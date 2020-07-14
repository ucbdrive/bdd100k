from __future__ import division

import re
from collections import OrderedDict

import torch
from mmcv.runner import Runner, MTLRunner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet import datasets
from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        Fp16OptimizerHook, BDDEvalHook, BddSegEvalHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger
import torch.distributed as dist


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        # TODO: fix reference frame has no gts
        elif loss_value == 0.0:
            continue
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode, step=-1):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def batch_processor_with_vis(model, data, train_mode, step=-1):
    # TEMPORARY
    with_vis = False # step%100 == 0
    losses = model(**data) # losses, vis
    loss, log_vars = parse_losses(losses)
    # gt label visualizations
    if with_vis:
        if 'gt_lane' in data:
            gt_lane = data['gt_lane'].data[0][0].permute(2,0,1).cpu().numpy()
            for i, g in enumerate(gt_lane):
                vis['vis_gt_lane_{}'.format(i)] = g
        if 'gt_drivable' in data:
            vis['vis_gt_drivable'] = data['gt_drivable'].data[0][0].cpu().numpy()
        if 'gt_sem_seg' in data:
            vis['vis_gt_sem_seg'] = data['gt_sem_seg'].data[0][0].cpu().numpy()
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data), img={'img': data['img'].data[0][0].cpu().numpy()}, vis=vis)
    else:
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None,
                   multitask=False,
                   vis=False):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if not vis:
        if distributed:
            _dist_train(model, dataset, cfg, validate=validate, multitask=multitask)
        else:
            _non_dist_train(model, dataset, cfg, validate=validate, multitask=multitask)
    else:
        if distributed:
            _dist_train(model, dataset, cfg, validate=validate, multitask=multitask, vis=vis)
        else:
            _non_dist_train(model, dataset, cfg, validate=validate, multitask=multitask, vis=vis)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        track_enhance = optimizer_cfg.pop('track_enhance', None)
        if track_enhance:
            params = []
            for name, param in model.named_parameters():
                if 'track' in name or 'asso' in name:
                    param_group = {
                        'params': [param],
                        'lr': track_enhance * optimizer_cfg['lr']
                    }
                else:
                    param_group = {'params': [param]}
                params.append(param_group)
            optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
            return optimizer_cls(params, **optimizer_cfg)
        else:
            return obj_from_dict(optimizer_cfg, torch.optim,
                                 dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False, multitask=False, vis=False):
    # prepare data loaders
    data_loaders = [[
        build_dataloader(
            d,
            cfg.data.imgs_per_gpu // 2 if issubclass(d.__class__, datasets.BDDVideo) else cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu // 2 if issubclass(d.__class__, datasets.BDDVideo) else cfg.data.workers_per_gpu,
            dist=True)
    for d in dataset]] if multitask else [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu // 2 if issubclass(dataset.__class__, datasets.BDDVideo) else cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu // 2 if issubclass(dataset.__class__, datasets.BDDVideo) else cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    bp = batch_processor_with_vis if vis else batch_processor
    runner = MTLRunner(model, bp, optimizer, cfg.work_dir,
                       cfg.log_level) if multitask else \
             Runner(model, bp, optimizer, cfg.work_dir,
                       cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            if not type(val_dataset_cfg) == list:
                val_dataset_cfg = [val_dataset_cfg]
            for _cfg in val_dataset_cfg:
                dataset_type = getattr(datasets, _cfg.type)
                if issubclass(dataset_type, datasets.BddStreet) or \
                   issubclass(dataset_type, datasets.BddSemanticSeg):
                    runner.register_hook(
                        BddSegEvalHook(_cfg, **eval_cfg))
                elif issubclass(dataset_type, datasets.CocoDataset):
                    runner.register_hook(
                        CocoDistEvalmAPHook(_cfg, **eval_cfg))
                elif issubclass(dataset_type, datasets.BDDVideo):
                    runner.register_hook(BDDEvalHook(_cfg))
                else:
                    runner.register_hook(
                        DistEvalmAPHook(_cfg, **eval_cfg))
            runner.register_logger_hooks(cfg.log_config)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if cfg.get('init_asso_head', False):
        ori_key = cfg.init_asso_head[0]
        new_key = cfg.init_asso_head[1]
        for _key in model.module.state_dict().keys():
            if 'asso' in _key:

                exist_key = _key.replace(ori_key, new_key)
                if exist_key in model.module.state_dict().keys():
                    if dist.get_rank() == 0:
                        print('Init "{}" with "{}"'.format(_key, exist_key))
                    model.module.state_dict()[_key].copy_(
                        model.module.state_dict()[exist_key])
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False, multitask=False, vis=False):
    # prepare data loaders
    data_loaders = [[
        build_dataloader(
            d,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    for d in dataset]] if multitask else [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    bp = batch_processor_with_vis if vis else batch_processor
    runner = MTLRunner(model, bp, optimizer, cfg.work_dir,
                    cfg.log_level) if multitask else \
             Runner(model, bp, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if cfg.get('init_asso_head', False):
        ori_key = cfg.init_asso_head[0]
        new_key = cfg.init_asso_head[1]
        for _key in model.module.state_dict().keys():
            if 'asso' in _key:
                exist_key = _key.replace(ori_key, new_key)
                if exist_key in model.module.state_dict().keys():
                    print('Init "{}" with "{}"'.format(_key, exist_key))
                    model.module.state_dict()[_key].copy_(
                        model.module.state_dict()[exist_key])
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
