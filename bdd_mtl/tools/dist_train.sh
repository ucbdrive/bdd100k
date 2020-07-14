#!/usr/bin/env bash
ROOT=/mnt/lustre/pangjiangmiao/jiangmiao/Detect-Track

CONFIG=$1
GPUS=$2
FOLDER=work_dirs

python3 -m torch.distributed.launch --nproc_per_node=$GPUS ⁠\
${ROOT}/tools/train.py \
./configs/${CONFIG}.py \
--work_dir=./${FOLDER}/${CONFIG} \
--launcher pytorch
