#!/bin/bash
# Run build.py for a list of clips sequentially on this machine's GPU: local_batch.sh <gpu> <clip> [clip ...]
G=$1; shift
cd /home/joeyw3/world_model/pvideo/pseudo
for c in "$@"; do
  CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 /shared/rsaas/joeyw3/miniconda3/envs/video/bin/python build.py clip $c > /shared/rsaas/joeyw3/pvideo_pseudo/logs/local_$c.log 2>&1
  echo "$c exit $? $(date)"
done
