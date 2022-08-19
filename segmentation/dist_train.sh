#!/usr/bin/env bash

export OMP_NUM_THREADS=1

CONFIG=$1
N_GPUS=$2
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
    --master_port=${PORT} \
    --use_env $(dirname "$0")/train.py ${CONFIG} --launcher pytorch ${@:3}

## bash dist_train.sh configs/sem_fpn_p2t_s_ade20k_80k.py 8

