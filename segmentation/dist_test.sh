#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29400}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:4}

## example command:
## bash dist_test.sh configs/sem_fpn_p2t_s_ade20k_80k.py pretrained/sem_fpn_p2t_s_ade20k_80k.pth 1
