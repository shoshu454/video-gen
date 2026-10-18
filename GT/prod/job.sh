#!/bin/bash
# Condor wrapper: job.sh <select|assign|render> <T> [args...]
source /shared/rsaas/joeyw3/miniconda3/etc/profile.d/conda.sh
conda activate qwen_video
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
cd /home/joeyw3/world_model/pvideo/prod
echo "host $(hostname) start $(date)"
python run.py "$@"
rc=$?
echo "exit $rc end $(date)"
exit $rc
