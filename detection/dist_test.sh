#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29400}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:4}

## example command:
## bash dist_test.sh configs/mask_rcnn_p2t_t_fpn_1x_coco.py pretrained/mask_rcnn_p2t_t_fpn_1x_coco-d875fa68.pth 1
