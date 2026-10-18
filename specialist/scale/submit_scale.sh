#!/bin/bash
# Overnight scale-up to 100 items per template (see worker.py / pipeline.py):
#   1. clips.json += scale clips; scale/work_A.txt, work_B.txt, fetch_B.txt   (make_clips_scale.py; old clips unchanged)
#   2. condor_submit jobA.sub (T1, T3, T4) and jobB.sub (T5, T6) on vision-h01 (vision-c22 has no schedd)
# Progress: <ROOT>/logs/scale/ (worker / pipeline / per-clip build logs), <ROOT>/qa/<T>/, <ROOT>/eval/summary_{A,B}.md
set -e
ROOT=/shared/rsaas/joeyw3/pvideo_pseudo
HERE=/home/joeyw3/world_model/pvideo/pseudo
mkdir -p $ROOT/logs/scale $ROOT/scale
cd $HERE
[ -f $ROOT/scale/work_A.txt ] || /shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python make_clips_scale.py --n_po ${N_PO:-200}
wc -l $ROOT/scale/work_A.txt $ROOT/scale/work_B.txt $ROOT/scale/fetch_B.txt
ssh -o BatchMode=yes vision-h01.cs.illinois.edu "cd $HERE/scale && condor_submit jobA.sub && condor_submit jobB.sub"
