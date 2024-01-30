#!/usr/bin/env bash

set -e

: ${DATA_DIR:=data2/cormac}
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/cormac_all.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
