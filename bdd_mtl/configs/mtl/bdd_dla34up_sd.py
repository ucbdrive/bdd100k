from bdd_mtl_factory import get_configs
import sys

################################################################################
#
# Format of the command to get configs:
#
#   $MODEL_NAME-$TASKS
#
#   task correspondence:
#     l - Lane marking
#     r - Drivable area
#     s - Semantic segmentation
#     d - Detection
#     i - Instance Segmentation
#     t - Multiple object Tracking
#     x - Multiple object Tracking with Segmentation
#
################################################################################

cfg = get_configs('dla34up-sd')

# Override default configs. Feel free to override more fields
cfg.optimizer['lr'] = 0.004
cfg.lr_config['step'] = [8, 11]
cfg.total_epochs = 20
cfg.data['imgs_per_gpu'] = 4
cfg.data['workers_per_gpu'] = 4
cfg.work_dir = './work_dirs/debug/mtl-sd'
cfg.load_from = './work_dirs/debug/mtl-s-lr01/latest.pth'
cfg.resume_from = None
cfg.checkpoint_config['interval'] = 10
cfg.workflow = [('train', 10)]

for k, v in cfg.__dict__.items():
    if not k.startswith('__'):
        setattr(sys.modules[__name__], k, v)
