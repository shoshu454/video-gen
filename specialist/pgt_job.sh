#!/bin/bash
# Condor wrapper: pgt_job.sh <clip name> [build.py flags]  -- runs the specialist models for one clip on one GPU.
source /shared/rsaas/joeyw3/miniconda3/etc/profile.d/conda.sh
conda activate video
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PGT_SCRATCH=/tmp/pgt_scratch_$$ HF_HOME=/home/joeyw3/.cache/huggingface
cd /home/joeyw3/world_model/pvideo/pseudo
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
# Some condor GPUs are occupied by processes outside condor: if the assigned GPU lacks memory, wait and ask condor
# to retry (exit 99, see max_retries in pgt.sub) instead of failing with an OOM.
FREE=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | grep "^${CUDA_VISIBLE_DEVICES%%,*}" | head -1 | cut -d, -f2 | tr -d " ")
if [ -z "$FREE" ] || [ "$FREE" -lt 38000 ]; then echo "GPU busy (free ${FREE} MiB), retry later"; sleep 600; exit 99; fi
python build.py clip "$@"
rc=$?
rm -rf /tmp/pgt_scratch_$$
echo "exit $rc end $(date)"
exit $rc
