# Stage A: fill in missing video geometry

```
raw dataset --adapter--> standard format --stage_a.py--> standard format with every modality present
```

Format spec, pitfalls and checks: **README_FORMAT.md**.

| file | what |
|---|---|
| `stage_a.py <in> <out>` | per video: GT modalities are hard-linked (or copied), missing ones are estimated with `../specialist` (`video` env, 1 GPU) |
| `stage_a.sub`, `stage_a.sh` | HTCondor: set `INPUT_DIR` / `OUTPUT_DIR` (and `NJOBS`) at the top, then `condor_submit stage_a.sub` from vision-h01 |
| `check_inputs.py <dir>` | validates an adapter's output (shapes, units, c2w vs w2c, static-point reprojection) |
| `check_outputs.py <out>` | verifies every `DONE` video has all modalities; lists unfinished videos and failures |
| `adapters/generic.py <src> <out> [--fps N]` | any .mp4 files / frame folders → RGB + meta.json |
| `adapters/TEMPLATE.py` | fill-in-the-blanks adapter for a dataset with GT |
| `adapters/pointodyssey.py` | worked example (PointOdyssey packed scenes) |
| `gdino_server.py` | Grounding DINO child process (`qwen_video` env, needs torch ≥ 2.4) used to name estimated masks |
| `tests/make_test_set.py <root>` | 6-video test set (2 full GT, 2 partial GT, 2 RGB only) |

## What gets estimated, and how

| missing | model (existing code in `../specialist`) | scale handling |
|---|---|---|
| intrinsics | MonST3R shared focal; refined by bundle adjustment when the extrinsics are estimated too | — |
| masks | SAM2.1 automatic instances + video propagation (`run_sam2_instances`); names from Grounding DINO boxes (box IoU ≥ 0.5 votes, otherwise `unknown`) | — |
| depth | Video Depth Anything metric ViT-L (`run_vda`) | with GT extrinsics: static CoTracker tracks are triangulated with the GT cameras and each depth map gets a smooth per-frame correction to them (`build.depth_correction`). VDA's metric scale was 2–4× off on PointOdyssey and drifted within a clip |
| extrinsics | MonST3R on ≤ 50 frames → metric scale from the depth (`build.cameras`) → bundle adjustment on the 2D tracks (`build.bundle_adjust`); `--pose_method vo`: PnP visual odometry (`build.vo_cameras`) | scaled to the (GT or estimated) depth; with GT `tracks_3d`: moved rigidly into their world frame |
| tracks | CoTracker3 online (`track_queries`): queries every 12 frames on a grid away from depth edges, ≥ 8 per mask instance, gaps filled near existing tracks; lifted with `build.lift_tracks` (static points get one world point, dynamic ones are smoothed) | — |

SAM2 and CoTracker run in segments (`--segment 300 --overlap 16`). SAM2 ids are matched across segments by IoU on
the overlap. Tracks that are still visible in the overlap are continued into the next segment. VDA windows the
video internally and MonST3R subsamples it, so both see the whole video at once.

## Resume and failures

- `<out>/<video>/DONE` is written last; re-running skips finished videos and finished modalities. Model outputs that
  later steps reuse (MonST3R, 2D tracks) are cached in `<out>/<video>/_work/` until the video is done.
- Every write is atomic: files go to a temp name and are renamed; modality folders are written as
  `<name>.tmp-<pid>/` and renamed into place.
- A failing video is logged to `<out>/failures.jsonl` (video, modality, error, traceback) and the run continues.
  `stage_a.py` exits 3 in that case. `stage_a.sh` maps that to 0, so condor doesn't retry deterministic
  failures, but does retry crashes and busy GPUs.

## Notes

- Peak GPU memory is a single model at a time (SAM2 ≈ 25 GiB is the largest), so one A40 is enough.
- `/shared/rsaas` is full and slow with small files. Put `OUTPUT_DIR` on `/home/joeyw3/...` or a local disk.
  The format stores one PNG per frame, so expect ~2 files per frame.
