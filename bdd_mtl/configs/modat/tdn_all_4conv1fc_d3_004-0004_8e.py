# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='TDN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=4,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    prop_track_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=801,
        featmap_strides=[4, 8, 16, 32]),
    prop_track_head=dict(
        type='PropRegTrackHead',
        num_shared_convs=4,
        num_shared_fcs=1,
        in_channels=801,
        fc_out_channels=1024,
        num_classes=2,
        reg_class_agnostic=True,
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    asso_track_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    asso_track_head=dict(
        type='AssoAppearTrackHead',
        num_convs=4,
        num_fcs=1,
        in_channels=256,
        roi_feat_size=7,
        fc_out_channels=1024,
        norm_cfg=norm_cfg,
        norm_similarity=False,
        loss_asso=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    freeze_exclude_tracker=False,
    corr_params=dict(
        patch_size=17, kernel_size=1, padding=0, stride=1, dilation_patch=1))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False),
    track=dict(asso_use_neg=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    track=dict(
        new_obj_score_thre=0.8,
        clean_before_short_assign=False,
        clean_before_long_assign=True,
        prop_overlap_thre=0.6,
        prop_score_thre=0.6,
        use_reid=True,
        long_term_frames=15,
        asso_score_thre=0.6,
        embed_momentum=0.5,
        prop_fn=False,
        update_score=True,
        plot_track_results=False))
# dataset settings
dataset_type = 'BDDVideo'
data_root = 'data/BDD/BDD_Tracking/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/bdd_tracking_train_0918.json',
        img_prefix=data_root + 'images/train/',
        img_scale=(1296, 720),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_track=True,
        with_crowd=True,
        with_label=True,
        train_sample_interval=3),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/mini_val_0918.json',
        img_prefix=data_root + 'images/val/',
        img_scale=(1296, 720),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_track=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/bdd_tracking_val_0918.json',
        img_prefix=data_root + 'images/val/',
        img_scale=(1296, 720),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        with_track=True))
# optimizer
optimizer = dict(
    type='SGD', lr=0.0004, momentum=0.9, weight_decay=0.0001, track_enhance=10)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_iters=2000,
    warmup_ratio=1.0 / 10.0,
    step=[5, 7])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'data/init_models/frcnn_r50_bdd100k_cls3_1x-65034c1b.pth'
# load_from = 'data/init_models/tdn_lr10-b3211c74.pth'
work_dir = './work_dirs/debug/'
resume_from = None
workflow = [('train', 1)]
