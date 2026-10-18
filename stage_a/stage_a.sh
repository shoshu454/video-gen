#!/bin/bash
# Condor wrapper of stage_a.py: stage_a.sh <input_dir> <output_dir> <shard i/n> [extra stage_a.py flags]
# Exit codes: 0 = all videos of the shard processed (per-video failures are in <output_dir>/failures.jsonl);
# 99 = assigned GPU busy, condor retries later; anything else = crash, condor retries (stage_a.py resumes).
# ---- site settings (see ../README.md): override them in the environment or edit the defaults here
CONDA_SH=${CONDA_SH:-/shared/rsaas/joeyw3/miniconda3/etc/profile.d/conda.sh}
VIDEO_ENV=${VIDEO_ENV:-video}                 # conda env with torch 2.1 + the specialist models
export PVIDEO_MODELS=${PVIDEO_MODELS:-/home/joeyw3}   # folder with Video-Depth-Anything/, monst3r/, co-tracker/
export PVIDEO_GDINO_PYTHON=${PVIDEO_GDINO_PYTHON:-/shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
MIN_FREE_MIB=${MIN_FREE_MIB:-38000}           # A40 (46 GB): wait if another process holds the GPU
# ----
IN=$1; OUT=$2; SHARD=$3; shift 3
source "$CONDA_SH"
conda activate "$VIDEO_ENV"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export PGT_SCRATCH=${TMPDIR:-/tmp}/stage_a_scratch_$$
cd "$(dirname "$(readlink -f "$0")")"
echo "host $(hostname) gpu '$CUDA_VISIBLE_DEVICES' shard $SHARD start $(date)"
# Some GPUs are used by processes outside condor: wait and retry if the assigned one lacks memory. Condor sets
# shortened UUIDs (GPU-0a12e7a8), so match them as a prefix of the full UUID.
FREE=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | grep "^${CUDA_VISIBLE_DEVICES%%,*}" | head -1 | cut -d, -f2 | tr -d " ")
if [ -z "$FREE" ] || [ "$FREE" -lt "$MIN_FREE_MIB" ]; then echo "GPU busy (free ${FREE} MiB), retry later"; sleep 600; exit 99; fi
python stage_a.py "$IN" "$OUT" --shard "$SHARD" --scratch "$PGT_SCRATCH" "$@"
rc=$?
rm -rf "$PGT_SCRATCH"
echo "exit $rc end $(date)"
python check_outputs.py "$OUT" --quick | tail -3
[ $rc -eq 3 ] && exit 0
exit $rc
