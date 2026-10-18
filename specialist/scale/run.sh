#!/bin/bash
# Condor wrapper of the scale-up jobs: run.sh <conda env> <script> [args...]   (cwd: the pseudo folder)
source /shared/rsaas/joeyw3/miniconda3/etc/profile.d/conda.sh
conda activate $1
shift
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg HF_HOME=/home/joeyw3/.cache/huggingface
cd /home/joeyw3/world_model/pvideo/pseudo
echo "host $(hostname) gpus '$CUDA_VISIBLE_DEVICES' start $(date): $*"
python "$@"
rc=$?
echo "exit $rc end $(date)"
exit $rc
