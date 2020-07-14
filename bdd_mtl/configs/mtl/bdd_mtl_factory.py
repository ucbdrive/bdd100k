from tasks import add_task_config

class BddMtlConfig:

    def __init__(self, model_name):
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.model = get_model_config(model_name)
        self.train_cfg = dict()
        self.test_cfg = dict()
        self.dataset_type = []
        self.data_root = 'data/bdd100k/'
        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.data = dict(
            imgs_per_gpu=2,
            workers_per_gpu=2,
            train=[],
            val=[],
            test=[])
        # optimizer
        self.optimizer = dict(
            type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) # , track_enhance=10)
        self.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
        # learning policy
        self.lr_config = dict(
            policy='step',
            warmup='exp',
            warmup_iters=500,
            warmup_ratio=0.1 / 3,
            step=[8, 11])
        self.checkpoint_config = dict(interval=1)
        # yapf:disable
        self.log_config = dict(
            interval=10,
            hooks=[
                dict(type='WandBLoggerHook', project_name='bdd-'),
                dict(type='TextLoggerHook')
                # dict(type='TensorboardLoggerHook')
            ])
        # yapf:enable
        # runtime settings
        self.total_epochs = 12
        self.dist_params = dict(backend='nccl')
        self.log_level = 'INFO'
        self.load_from = None
        self.work_dir = './work_dirs/debug/bdd'
        self.resume_from = None
        self.workflow = [('train', 1)]

# backbones

def get_model_config(model_name):

    if model_name == 'dla34up':
        model_config = dict(
            type='MTL',
            pretrained='weights/dla34-ba72cf86.pth',
            backbone=dict(
                type='DLA',
                levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block_num=2,
                return_levels=True),
            neck=dict(
                type='DLAUp',
                in_channels=[32, 64, 128, 256, 512],
                channels=[256, 256, 256, 256, 256],
                scales=(1, 2, 4, 8, 16),
                num_outs=5)
        )
    elif model_name == 'dla34':
        model_config = dict(
            type='MTL',
            pretrained='weights/dla34-ba72cf86.pth',
            backbone=dict(
                type='DLA',
                levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block_num=2,
                return_levels=True)
        )
    elif model_name == 'dla60up':
        model_config = dict(
            type='MTL',
            pretrained='weights/dla60-24839fc4.pth',
            backbone=dict(
                type='DLA',
                levels=[1, 1, 1, 2, 3, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block_num=1,
                return_levels=True),
            neck=dict(
                type='DLAUp',
                in_channels=[16, 32, 128, 256, 512, 1024],
                channels=[256, 256, 256, 256, 256],
                scales=(1, 2, 4, 8, 16),
                num_outs=5)
        )
    elif model_name == 'dla102up':
        model_config = dict(
            type='MTL',
            pretrained='weights/dla60-24839fc4.pth',
            backbone=dict(
                type='DLA',
                levels=[1, 1, 1, 3, 4, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block_num=1,
                return_levels=True),
            neck=dict(
                type='DLAUp',
                in_channels=[16, 32, 128, 256, 512, 1024],
                channels=[256, 256, 256, 256, 256],
                scales=(1, 2, 4, 8, 16),
                num_outs=5)
        )
    elif model_name == 'res50':
        model_config = dict(
            type='MTL',
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
        )
    elif model_name == 'res101':
        model_config = dict(
            type='MTL',
            pretrained='modelzoo://resnet101', # 'weights/dla34-ba72cf86.pth',
            backbone=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                style='pytorch'),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
        )
    return model_config

def get_configs(mtl_command):
    model_name, tasks = mtl_command.split('-')
    cfg = BddMtlConfig(model_name)
    cfg = add_task_config(tasks, cfg)
    return cfg
