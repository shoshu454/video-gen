# Pseudo-GT: running the QA pipeline on videos without ground truth

The production pipeline (`../prod/run.py` + the `../walkthrough_*.py` selection and label code) reads dataset GT:
PointOdyssey (T1, T3, T4, T5) and KITTI raw (T6). This folder replaces that GT with specialist-model estimates.
It writes them **in the exact on-disk formats of the GT**, so the pipeline runs unmodified. Only the data and
output roots are redirected (`run_pseudo.py`).

## GT → specialist mapping

| GT the pipeline reads | where it is used | specialist / derivation |
|---|---|---|
| PO `depths/*.png` (z, 16-bit, ×1000/65535 m) | patch σ/μ, depth-agreement, occlusion confirmation (T3) | **Video Depth Anything, metric ViT-L**, made consistent with the fused static geometry (per-frame log-plane correction + multi-view temporal median) |
| PO `intrinsics`, `extrinsics` (world→cam) | camera centres / speed, P_cam, r | **MonST3R** global alignment on ≤50 strided frames, then **bundle adjustment** of all poses + focal on the dense tracks, slerp/lerp to all frames, scale aligned to VDA metric depth |
| PO `trajs_2d`, `visibs`, `valids` | every point rule | **CoTracker3** (online model, both directions), 12 px grid at frame 0 + fill-in queries every 1 s on uncovered areas |
| PO `trajs_3d` (world) → `P_cam`, `r`, `dyn` | all labels | CoTracker tracks lifted with VDA depth + MonST3R cameras. A track is dynamic if no static 3D point along its ray reprojects onto it within 4 px (80th pct). Static tracks get one world point; dynamic tracks are robustly smoothed and interpolated through occlusions |
| PO `masks/*.png` (instance ids) | region rules (area, presence, border, background) | **SAM2.1** automatic masks (object-level picking) at frame 0 + new objects every 2 s, video propagation forward and backward |
| KITTI OXTS (pose → speed, yaw rate → stop/turn events) | T6 events, ego-pose propagation | **metric PnP visual odometry** (VDA depth at t + CoTracker matches at t+1..t+3, RANSAC, traffic participants masked), encoded back into OXTS lat/lon/alt/rpy + smoothed vn/ve/wu |
| KITTI Velodyne scans | z-buffer visibility, patch σ/μ, round trip, R_MAX | **pseudo-LiDAR** from VDA metric depth (every 2nd pixel, z < 80 m, HDL-64E vertical FOV −24.9°..+2°) |
| KITTI calibration | projection | MonST3R focal; `P_rect_02 = [K|0]`, `R_rect_00 = I`, velodyne = KITTI axes at the camera centre |
| KITTI `tracklet_labels.xml` | exclude points on objects | **Grounding DINO** (car/van/truck/bus/person/bicycle/…) boxes → **SAM2.1** masks → lifted → oriented 3D box per frame |

## Files

- `specialists.py`: model wrappers (`video` env, GPU).
- `build.py`: runs the specialists for a clip or any video, fuses the outputs, and writes the PO or KITTI layout.
  Raw outputs are cached in `/shared/rsaas/joeyw3/pvideo_pseudo/raw/<name>.npz`. Grounding DINO needs
  torch ≥ 2.4, so it runs in the `qwen_video` env as a subprocess.
- `run_pseudo.py`: `prod/run.py` with redirected roots (`qwen_video` env). Same commands: select / assign / render.
- `make_clips.py`: the evaluation clips (10 GT items per template, each window ±2 s; KITTI ±3 s).
- `eval_gt.py`: **A**) re-answers each GT item from pseudo-GT (its marked points are tracked as CoTracker probes);
  **B**) maps every generated item back to GT and re-labels it with the GT data and the same rules.
- `diag.py`: per-component metrics of a pseudo scene vs PO GT (depth, camera, tracks, motion labels, masks).
- `pgt.sub`, `pgt_job.sh`: condor (A40) jobs, one clip each. `local_batch.sh` runs clips on a local GPU.

Data roots: PO-format scenes are in `/shared/rsaas/joeyw3/pvideo_pseudo/data/po/pseudo/<name>`. KITTI-format drives
are in `/tmp/claude-1267708/pgt_kitti` on local disk (one OXTS file per frame; NFS is slow with small files). They
are rebuilt from the raw cache with `build.py clip <name> --fuse_only`. Items go to `/shared/rsaas/joeyw3/pvideo_pseudo/qa/<T>/` (`items/` holds two early test renders).

## Usage on any video

```bash
# PO layout (T1/T3/T4/T5); the video is resampled to 24 fps because common.load_po hard-codes it
python build.py video my.mp4 --name my_clip                     # video env, 1 GPU
python run_pseudo.py select T3 pseudo/my_clip                    # qwen_video env
python run_pseudo.py assign T3 --n 10 && python run_pseudo.py render T3 <assign.json> 1,2,...
# KITTI layout (T6, driving video)
python build.py video drive.mp4 --name my_drive --kitti          # prints the drive id
python run_pseudo.py select T6 <drive id>
```

## Changes relative to GT behaviour (all outside the pipeline code)

- `run_pseudo.PSEUDO_TOL`: `DEPTH_AGREE` 0.02 → 0.05 and `STD_MAX` 0.02 → 0.03 for PO templates. These are applied
  the same way `prod/run.py` overrides walkthrough constants. Pseudo depth and lifted tracks agree to ~1.2% median
  per frame. Requiring ≤2% on every frame of a window removed every T1 candidate (funnel on T3_001: 26 → 0;
  with 5%: 458 → 340; GT: 716 → 508).
- `run_pseudo.SEARCH_PATCH`: T1 and T3 verify only the top-scored candidate per window. On pseudo-GT the extreme
  candidates are enriched for estimation artifacts, so every window was used up (T1_001: 11/11 windows failed,
  0 items). Up to 5 (T1) / 4 (T3 per kind) candidates are tried per window, still at most one kept per window.
- MonST3R (`dust3r/cloud_opt/optimizer.py:263`): numpy instead of torch indices. The torch-index path was >100× slower.
- Every script calls `prctl(PR_SET_THP_DISABLE)`. On vision-c22, transparent-hugepage compaction stalls make
  large CPU allocations take minutes.

## Known limits (measured on PO / KITTI GT, see eval/)

- Dynamic points: CoTracker on PO's fast-moving animated characters has a ~9 px median error (static points: 0.6 px).
- VDA metric depth on PO: AbsRel ~0.15 and a global scale ~1.3× GT. Templates use ratios, so the scale doesn't
  matter there. On KITTI the ego speed is ~1.3× too high (depth scale and focal), which shifts the stop-event thresholds.
- MonST3R was trained on PointOdyssey *train*, so PO-train scenes are in-distribution for its cameras.
- The `agent` motion statistic (max speed over all dynamic tracks) is ~2x GT on T3_001 (median 5.7 vs 2.6 m/s),
  even after Hampel filtering, splitting tracks at depth jumps, and the far/border static rules. So the
  moving-agent window predicate passes somewhat more often than with GT.
