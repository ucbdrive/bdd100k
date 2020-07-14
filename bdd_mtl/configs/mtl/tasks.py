
def add_task_config(tasks, cfg):
    # modules
    if 'c' in tasks:
        cfg = add_cls_modules(cfg)
    if 'l' in tasks:
        cfg = add_lane_modules(cfg)
    if 'r' in tasks:
        cfg = add_drivable_modules(cfg)
    if 's' in tasks:
        cfg = add_sem_seg_modules(cfg)
    if len(set(tasks) & set('ditx')) > 0:
        # if only detection is present, use 10 classes
        if set(tasks) & set('ditx') == set(['d']):
            cfg = add_box_modules(cfg, num_classes=11)
        else:
            cfg = add_box_modules(cfg)
    # # TEMPORARY
    # cfg = add_box_modules(cfg, num_classes=11)
    if len(set(tasks) & set('tx')) > 0:
        cfg = add_track_modules(cfg)
    if len(set(tasks) & set('ix')) > 0:
        cfg = add_mask_modules(cfg)

    # datasets
    cfg_add_data_for_task = dict(
        c=add_cls_data,
        l=add_lane_data,
        r=add_driv_data,
        s=add_sem_seg_data,
        d=add_det_data,
        i=add_ins_seg_data,
        t=add_box_track_data,
        x=add_seg_track_data
    )

    if 'l' in tasks and 'r' in tasks:
        # lane marking and drivable area
        cfg = add_lane_driv_data(cfg)
        tasks = [t for t in tasks if t not in 'lr']

    for t in tasks:
        if t in cfg_add_data_for_task:
            cfg = cfg_add_data_for_task[t](cfg)
    return cfg

# modules
# classification
def add_cls_modules(cfg):
    cfg.model.update(dict(
        cls_head=dict(
            type='ClsHead',
            num_convs=2,
            in_channels=512,
            conv_kernel_size=3,
            conv_out_channels=256,
            num_classes=[6, 6, 3],
            conv_cfg=None,
            norm_cfg=None,
            loss_cls=dict(
                type='CrossEntropyLoss',
                ignore_index=-1)
        )
    ))
    return cfg

# lane marking
def add_lane_modules(cfg):
    cfg.model.update(dict(
        lane_dir_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            name='lane_dir',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear'),
        lane_sty_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            name='lane_sty',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear'),
        lane_typ_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            name='lane_typ',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            fg_weight=10,
            upsample_method='bilinear')
    ))

    return cfg

# drivable area
def add_drivable_modules(cfg):
    cfg.model.update(dict(
        drivable_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            name='drivable',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            upsample_method='bilinear')
    ))
    return cfg


# semantic segmentation
def add_sem_seg_modules(cfg):
    # model settings
    cfg.model.update(dict(
        sem_seg_head=dict(
            type='MultiLabelFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=19,
            name='sem_seg',
            loss_mask=dict(
                type='CrossEntropyLoss', ignore_index=255),
            upsample_method='bilinear')
    ))
    return cfg

# detection
def add_box_modules(cfg, num_classes=9):
    # model settings
    cfg.model.update(dict(
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
            num_classes=num_classes,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ))
    # model training and testing settings
    cfg.train_cfg.update(dict(
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
            debug=False)))
    cfg.test_cfg.update(dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    ))

    return cfg

def add_mask_modules(cfg):
    cfg.model.update(dict(
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ))
    cfg.train_cfg.update(dict(
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
            mask_size=28,
            pos_weight=-1,
            debug=False)))
    cfg.test_cfg.update(dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    ))

    return cfg

# tracking
def add_track_modules(cfg):
    # model settings
    cfg.model.update(dict(
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
            norm_cfg=cfg.norm_cfg,
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
            norm_cfg=cfg.norm_cfg,
            norm_similarity=False,
            loss_asso=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        freeze_exclude_tracker=False,
        corr_params=dict(
            patch_size=17, kernel_size=1, padding=0, stride=1, dilation_patch=1)
    ))

    # model training and testing settings
    cfg.train_cfg.update(dict(
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
            track=dict(asso_use_neg=False)
    ))
    cfg.test_cfg.update(dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
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
            update_score=False,
            plot_track_results=False)
    ))
    return cfg

# datasets
# classification
def add_cls_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['BddCls'])
    cfg.data['train'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/train/',
            label_dir=cfg.data_root+'labels/cls/cls_train.json',
            phase='mini_train',
            flip_ratio=0.5,
            img_prefix=cfg.data_root+'images/100k/train/',
            img_scale=(1280, 720),
            crop_size=(640, 640),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            ann_file=None
        )
    )
    cfg.data['val'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/cls/cls_val.json',
            phase='val',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            test_mode=True,
            ann_file=None
        )
    )
    cfg.data['test'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/cls/cls_val.json',
            phase='test',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            test_mode=True,
            ann_file=None
        )
    )
    # output settings
    cfg.work_dir += 'c'
    cfg.log_config['hooks'][0]['project_name'] += 'c'
    return cfg

# lane marking
def add_lane_data(cfg):
    return add_lane_driv_data(cfg, True, False)

# drivable area
def add_driv_data(cfg):
    return add_lane_driv_data(cfg, False, True)

# lane marking and drivable area
def add_lane_driv_data(cfg, with_lane=True, with_driv=True):
    # dataset settings
    cfg.dataset_type.append(['BddStreet'])
    cfg.data['train'].append(
        dict(type='BddStreet',
            image_dir=cfg.data_root+'images/100k/train/',
            label_dir=cfg.data_root+'labels/street/street_train',
            phase='train',
            flip_ratio=0.5,
            with_lane=with_lane,
            with_drivable=with_driv,
            img_prefix=cfg.data_root+'images/100k/train/',
            img_scale=(1280, 720),
            crop_size=(640, 640),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None
        )
    )
    cfg.data['val'].append(
        dict(type='BddStreet',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/street/street_val',
            phase='mini_val',
            flip_ratio=0,
            with_lane=with_lane,
            with_drivable=with_driv,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None
        )
    )
    cfg.data['test'].append(
        dict(type='BddStreet',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/street/street_val',
            phase='val',
            flip_ratio=0,
            with_lane=with_lane,
            with_drivable=with_driv,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None
        )
    )
    # output settings
    if with_lane:
        cfg.work_dir += 'l'
        cfg.log_config['hooks'][0]['project_name'] += 'l'
    if with_driv:
        cfg.work_dir += 'r'
        cfg.log_config['hooks'][0]['project_name'] += 'r'
    return cfg

# semantic segmentation
def add_sem_seg_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['BddSemanticSeg'])
    cfg.data['train'].append(
        dict(type='BddSemanticSeg',
            image_dir=cfg.data_root+'images/10k/train/',
            label_dir=cfg.data_root+'labels/sem_seg/sem_seg_train',
            phase='train',
            flip_ratio=0.5,
            img_prefix=cfg.data_root+'images/10k/train/',
            img_scale=(1280, 720),
            crop_size=(640, 640),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None
        )
    )
    cfg.data['val'].append(
        dict(type='BddSemanticSeg',
            image_dir=cfg.data_root+'images/10k/val/',
            label_dir=cfg.data_root+'labels/sem_seg/sem_seg_val',
            phase='val',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/10k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None,
            test_mode=True
        )
    )
    cfg.data['test'].append(
        dict(type='BddSemanticSeg',
            image_dir=cfg.data_root+'images/10k/val/',
            label_dir=cfg.data_root+'labels/sem_seg/sem_seg_val',
            phase='val',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/10k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            with_crowd=True,
            with_label=True,
            ann_file=None,
            test_mode=True
        )
    )
    # output settings
    cfg.work_dir += 's'
    cfg.log_config['hooks'][0]['project_name'] += 's'
    return cfg

# detection
def add_det_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['CocoDataset'])
    det_prefix = 'det_8cls' if cfg.model['bbox_head']['num_classes'] == 9 else 'det'
    cfg.data['train'].append(
        dict(
            type='CocoDataset',
            ann_file=cfg.data_root+'labels/det/coco_format/{}_train.json'.format(det_prefix),
            img_prefix=cfg.data_root+'images/100k/train/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True)
    )

    cfg.data['val'].append(
        dict(
            type='CocoDataset',
            ann_file=cfg.data_root+'labels/det/{}_val.json'.format(det_prefix),
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=True,
            with_label=True,
            task='det')
    )

    cfg.data['test'].append(
        dict(
            type='CocoDataset',
            ann_file='/shared/haofeng/bdd100k/labels/det/{}_val.json'.format(det_prefix),
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True,
            task='det')
    )
    # output settings
    cfg.work_dir += 'd'
    cfg.log_config['hooks'][0]['project_name'] += 'd'
    return cfg

# instance segmentation
def add_ins_seg_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['CocoDataset'])
    cfg.data['train'].append(
        dict(
            type='CocoDataset',
            ann_file=cfg.data_root+'labels/ins_seg/ins_seg_train.json',
            img_prefix=cfg.data_root+'images/10k/train/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=True,
            with_crowd=True,
            with_label=True)
    )
    cfg.data['val'].append(
        dict(
            type='CocoDataset',
            ann_file=cfg.data_root+'labels/ins_seg/ins_seg_val.json',
            img_prefix=cfg.data_root+'images/10k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=True,
            with_crowd=True,
            with_label=True,
            task='ins_seg')
    )
    cfg.data['test'].append(
        dict(
            type='CocoDataset',
            ann_file=cfg.data_root+'labels/ins_seg/ins_seg_val.json',
            img_prefix=cfg.data_root+'images/10k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=True,
            with_crowd=True,
            with_label=True,
            task='ins_seg')
    )
    # output settings
    cfg.work_dir += 'i'
    return cfg

# box tracking
def add_box_track_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['BDDVideo'])
    cfg.data['train'].append(
        dict(
            type='BDDVideo',
            ann_file='/data3/haofeng/waymo/labels/waymo12_front_train_3cls.json', # cfg.data_root+'labels/box_track/box_track_train.json',
            img_prefix='/data3/haofeng/waymo/images', # cfg.data_root+'images/tracking/train/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_track=True,
            with_crowd=True,
            with_label=True,
            train_sample_interval=3)
    )
    cfg.data['val'].append(
        dict(
            type='BDDVideo',
            ann_file='/data3/haofeng/waymo/labels/waymo12_front_val_3cls.json', # cfg.data_root+'labels/box_track/box_track_mini_val.json',
            img_prefix='/data3/haofeng/waymo/images', # cfg.data_root+'images/tracking/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=True,
            with_label=True,
            with_track=True,
            task='box_track')
    )
    cfg.data['test'].append(
        dict(
            type='BDDVideo',
            ann_file= '/shared/haofeng/bdd-tracking-2k/labels/bdd100k_track_val_0519.json', # '/shared/haofeng/bdd-tracking-2k/labels/bdd-tracking-2k_8cls_val_0216.json', # '/data3/haofeng/waymo/labels/waymo12_front_val_3cls.json', # cfg.data_root+'labels/box_track/box_track_val.json',
            img_prefix='/shared/haofeng/bdd-tracking-2k/images/val', # cfg.data_root+'images/tracking/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True,
            with_track=True,
            task='box_track')
    )
    # output settings
    cfg.work_dir += 't'
    cfg.log_config['hooks'][0]['project_name'] += 't'
    return cfg

# seg tracking
def add_seg_track_data(cfg):
    # dataset settings
    cfg.dataset_type.append(['BDDVideo'])
    cfg.data['train'].append(
        dict(
            type='BDDVideo',
            ann_file=cfg.data_root+'labels/seg_track/seg_track_train_legacy.json',
            img_prefix='/shared/haofeng/bdd100k/images/tracking/train/', # '/shared/haofeng/bdd-tracking-2k/images/train/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=True,
            with_track=True,
            with_crowd=True,
            with_label=True,
            train_sample_interval=3)
    )
    cfg.data['val'].append(
        dict(
            type='BDDVideo',
            ann_file=cfg.data_root+'labels/seg_track/seg_track_val_legacy.json',
            img_prefix='/shared/haofeng/bdd100k/images/tracking/val/', # '/shared/haofeng/bdd-tracking-2k/images/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=True,
            with_label=False,
            test_mode=True,
            with_track=True,
            task='seg_track')
    )
    cfg.data['test'].append(
        dict(
            type='BDDVideo',
            ann_file=cfg.data_root+'labels/seg_track/seg_track_val_new.json',
            img_prefix='/shared/haofeng/bdd-tracking-2k/images/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=True,
            with_label=False,
            test_mode=True,
            with_track=True,
            task='seg_track')
    )
    # output settings
    cfg.work_dir += 'x'
    cfg.log_config['hooks'][0]['project_name'] += 'x'
    return cfg
