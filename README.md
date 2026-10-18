# pvideo

Fills in the 3D geometry of ordinary videos: per-frame depth, instance masks, camera intrinsics, camera poses and
point tracks (2D + 3D). It uses existing specialist models (Video Depth Anything, MonST3R, SAM2.1, CoTracker3,
Grounding DINO). You give it a folder of videos; you get back one folder per video in a fixed format.

```
pvideo/
  stage_a/      <- what you run (start here)
  specialist/   model wrappers + fusion code that stage_a calls (specialists.py, build.py)
  GT/           the original question-generation pipeline that reads dataset GT (PointOdyssey, KITTI); reference only
```

`GT/` and `specialist/` also contain the scripts of our own PointOdyssey / KITTI experiments. Those scripts hard-code
paths on our cluster and are **not** needed to run Stage A. Stage A uses only `specialist/specialists.py` and
`specialist/build.py`.

---

## 1. Hardware

- Linux with an NVIDIA GPU. **Only tested on NVIDIA A40 (46 GB)** with CUDA 12.1 PyTorch builds, under HTCondor.
- One GPU per job. The models run one after another; the largest peak is SAM2 at about 25 GiB, plus about 6 GiB of
  weights kept loaded.
- About 80 GB CPU RAM per job (the whole video is held in memory), and about 10 GB of disk for checkpoints.
- Speed on one A40: about 7–14 minutes for a 100–200 frame clip when everything has to be estimated, plus about
  5 minutes of model loading once per job.

**Using a different GPU takes extra work:**
- In `stage_a/stage_a.sub`, change `requirements = (GPUs_DeviceName == "NVIDIA A40")`.
- In `stage_a/stage_a.sh`, change `MIN_FREE_MIB` (the job waits until this much GPU memory is free).
- With less than ~40 GB of GPU memory, lower `--segment` (e.g. 150), `--cotracker_max_side` (e.g. 640) and
  `--monst3r_frames` (e.g. 30) through `EXTRA_ARGS`. This is untested.
- Newer GPUs (e.g. Blackwell) need a newer PyTorch than 2.1. That means rebuilding the `video` environment and
  re-checking that MonST3R, SAM2 and Video Depth Anything still work with it. This is untested.

## 2. What to clone

Put the three model repositories in one folder. Its path is called `$PVIDEO_MODELS` below.

```bash
export PVIDEO_MODELS=/path/to/models          # any folder
cd $PVIDEO_MODELS
git clone https://github.com/DepthAnything/Video-Depth-Anything && git -C Video-Depth-Anything checkout 4f5ae23
git clone --recursive https://github.com/junyi42/monst3r            && git -C monst3r checkout 574cc77
git clone https://github.com/facebookresearch/co-tracker              && git -C co-tracker checkout 82e02e8
git clone <this repo> pvideo                                           # anywhere
```

The commits are the ones we tested. Newer ones may work.

**Required MonST3R patch** (without it, flow precomputation is >100× slower):
```bash
cd $PVIDEO_MODELS/monst3r
sed -i 's/np.stack(self.imgs)\[self._ei\], np.stack(self.imgs)\[self._ej\]/np.stack(self.imgs)[np.asarray(self._ei)], np.stack(self.imgs)[np.asarray(self._ej)]/' dust3r/cloud_opt/optimizer.py
grep -n "np.asarray(self._ei)" dust3r/cloud_opt/optimizer.py    # must print one line (around line 263)
```

## 3. Python environments

You need two conda environments, because Grounding DINO (via `transformers`) needs torch ≥ 2.4 and the other models
were tested with torch 2.1.

**`video`** runs everything except Grounding DINO. Tested versions: Python 3.10, torch 2.1.1+cu121,
torchvision 0.16.1, numpy 1.24, scipy 1.14, opencv 4.10, av 17, imageio 2.37, hydra-core 1.3.2, xformers 0.0.23.
```bash
conda create -n video python=3.10 -y && conda activate video
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r $PVIDEO_MODELS/monst3r/requirements.txt
pip install -r $PVIDEO_MODELS/Video-Depth-Anything/requirements.txt
pip install -e $PVIDEO_MODELS/co-tracker
pip install -e $PVIDEO_MODELS/monst3r/third_party/sam2
pip install "numpy<2" scipy opencv-python av imageio hydra-core iopath
```

**`gdino`** runs Grounding DINO only. Tested: Python 3.10, torch 2.12 (cu130), transformers 5.10.
```bash
conda create -n gdino python=3.10 -y && conda activate gdino
pip install torch torchvision                       # any build >= 2.4 that matches your driver
pip install transformers numpy pillow
```

These install commands rebuild what we used. We did not test them from scratch on a clean machine. If an import
fails, match the tested versions above.

## 4. Model checkpoints

| model | used for | file (path relative to `$PVIDEO_MODELS`) | how to get it |
|---|---|---|---|
| Video Depth Anything, metric ViT-L | depth | `Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth` (1.5 GB) | `wget -P Video-Depth-Anything/checkpoints https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth` |
| MonST3R | camera poses + focal | `monst3r/checkpoints/{config.json, model.safetensors}` (2.3 GB) | `huggingface-cli download Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt --local-dir monst3r/checkpoints` |
| SEA-RAFT / RAFT | optical flow inside MonST3R | `monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth` (+ the `raft-*.pth`) | `cd monst3r/data && bash download_ckpt.sh` (also fetches the MonST3R and SAM2 weights; needs `pip install gdown`) |
| SAM2.1 hiera-large | instance masks | `monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt` (0.9 GB) | `wget -P monst3r/third_party/sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |
| CoTracker3 online | point tracks | `co-tracker/checkpoints/scaled_online.pth` (0.1 GB) | `wget -P co-tracker/checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth` |
| Grounding DINO base | names for the masks | Hugging Face cache (`$HF_HOME`) | downloaded on first use; pre-download it (compute nodes often have no internet): `conda activate gdino && python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection as M; AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); M.from_pretrained('IDEA-Research/grounding-dino-base')"` |

## 5. Point the code at your installation

Set these at the top of `stage_a/stage_a.sh`. Condor jobs don't inherit your shell, so set them there as the
defaults, not only in your shell. When running `stage_a.py` by hand, export them instead.

| variable | meaning | example |
|---|---|---|
| `CONDA_SH` | conda's activation script | `$HOME/miniconda3/etc/profile.d/conda.sh` |
| `VIDEO_ENV` | name of the `video` env | `video` |
| `PVIDEO_MODELS` | folder with the three cloned model repos | `/data/me/models` |
| `PVIDEO_GDINO_PYTHON` | python of the `gdino` env | `$HOME/miniconda3/envs/gdino/bin/python` |
| `HF_HOME` | Hugging Face cache holding Grounding DINO | `$HOME/.cache/huggingface` |
| `MIN_FREE_MIB` | GPU memory that must be free before the job starts | `38000` (A40) |

## 6. Input: a folder of videos

The **only** accepted input is one folder that contains:
- video files (`.mp4 .mov .mkv .avi .webm .m4v`), and/or
- frame folders: subfolders of `.png` / `.jpg` / `.jpeg` images, ordered by file name.

```
my_videos/
  cooking_01.mp4
  street.mov
  robot_run_3/            frame folder
    000001.jpg
    000002.jpg ...
```

Video names are taken from the file / folder names, so they must be unique.

Step 1 converts the folder to the standard format. This takes seconds per video and needs no GPU. `.mp4` files are
hard-linked; other containers and frame folders are written as PNG frames:
```bash
conda activate video
cd pvideo/stage_a
python adapters/generic.py /path/to/my_videos /path/to/std_input --fps 30   # --fps only matters for frame folders
python check_inputs.py /path/to/std_input                                    # every line should say [OK]
```
The frame rate of a video file is read from the file. `--fps` applies to every frame folder: put frame folders with
different rates in separate input folders.

If you already have GT (depth, masks, cameras, tracks) for your videos, write a dataset adapter instead
(`adapters/TEMPLATE.py`, spec in `stage_a/README_FORMAT.md`). Stage A keeps your GT and estimates only what is missing.

## 7. Run

**On HTCondor** (from a submit node; at our site that is vision-h01). Edit the top of `stage_a/stage_a.sub`:
```
INPUT_DIR  = /path/to/std_input       # the folder from step 6
OUTPUT_DIR = /path/to/stage_a_output  # created if missing; must not be inside INPUT_DIR
NJOBS      = 4                        # parallel GPU jobs; each takes every NJOBS-th video
EXTRA_ARGS =                          # optional stage_a.py flags, e.g. --segment 150
```
```bash
cd pvideo/stage_a
condor_submit stage_a.sub
condor_q                                   # job state
tail -f logs/stage_a_<cluster>_0.out       # progress
```

**On a machine with a GPU, without condor:**
```bash
conda activate video && cd pvideo/stage_a
export PVIDEO_MODELS=... PVIDEO_GDINO_PYTHON=...
CUDA_VISIBLE_DEVICES=0 python stage_a.py /path/to/std_input /path/to/stage_a_output
```

**Afterwards:**
```bash
python check_outputs.py /path/to/stage_a_output   # [OK] per finished video, unfinished videos, failures
```

- **Resume:** running the same command again, or resubmitting the same `.sub`, continues where it stopped. Finished
  videos have a `DONE` file and are skipped; finished modalities inside a video are kept.
- **Failures:** a video that fails is skipped, and the error and traceback are appended to
  `OUTPUT_DIR/failures.jsonl`. Fix the cause and resubmit.
- **Storage:** put `OUTPUT_DIR` on a disk that handles many small files well. The format writes 2 PNGs per frame.

## 8. Output

One folder per video. Every modality is always present:

```
stage_a_output/
  failures.jsonl                 only if something failed
  <video_name>/
    video.mp4 | rgb/00000.png...  the input frames, unchanged
    meta.json                     see below
    depth/00000.png ...           uint16 PNG, z-depth along the optical axis in millimetres, 0 = invalid / > 65.5 m
    masks/00000.png ...           uint16 PNG, instance ids, consistent over time, 0 = background
    masks/names.json              {"<id>": "person" | ... | "unknown"}
    tracks/tracks_2d.npy          (T, N, 2) float32, pixel x, y
    tracks/visibility.npy         (T, N) bool
    tracks/query_frame.npy        (N,) int64, frame each track started from
    tracks/tracks_3d.npy          (T, N, 3) float32, world coordinates in metres
    camera/intrinsics.npy         (3, 3) float32 K in pixels
    camera/extrinsics.npy         (T, 4, 4) float32 camera-to-world, OpenCV axes (x right, y down, z forward), metres
    DONE                          written last
```

`meta.json`:
```json
{
  "dataset": "generic", "source_path": "/path/to/my_videos/street.mov", "fps": 30.0,
  "num_frames": 240, "resolution": [1280, 720],
  "sources": {"intrinsics": "monst3r+bundle_adjustment", "masks": "sam2.1_auto", "depth": "video_depth_anything_metric_vitl",
              "extrinsics": "monst3r+bundle_adjustment", "tracks": "cotracker3_online+lifted", "mask_names": "grounding_dino"},
  "scale_alignment": {"extrinsics": {"factor": 21.3, "to": "estimated depth", "method": "..."}},
  "stage_a": {"version": "stage_a 1.0", "host": "...", "finished": "2026-10-18T00:28:43"}
}
```

`sources` says, per modality, whether it is `gt` (copied from the input) or which model estimated it.
`scale_alignment` records the scale factors used to make depth and camera translation agree. Full specification:
`stage_a/README_FORMAT.md`.

### `stage_a.py` options

| flag | default | meaning |
|---|---|---|
| `--segment`, `--overlap` | 300, 16 | SAM2 / CoTracker work on segments of this many frames, stitched on the overlap |
| `--query_every` | 12 | new CoTracker query points every N frames |
| `--track_spacing` | 12 px per 960 px | query grid spacing |
| `--cotracker_max_side` | 960 | CoTracker input resolution (long side) |
| `--monst3r_frames` | 50 | frames used by MonST3R global alignment |
| `--pose_method` | `monst3r` | `vo`: PnP visual odometry instead (driving videos) |
| `--vocab` | ~40 common objects | Grounding DINO words used to name masks |
| `--no_names` | off | skip Grounding DINO (masks stay unnamed) |
| `--videos a,b` | all | process only these videos |
| `--shard i/n` | `0/1` | process every n-th video starting at i (set by `stage_a.sub`) |

## 9. Accuracy and limits (measured on PointOdyssey)

- Camera poses estimated with GT depth: path length within 3%, trajectory error about 1 cm over 90 frames.
- Depth estimated with GT cameras: within about 4% of GT after alignment.
- **RGB-only videos have no metric reference.** Depth comes from Video Depth Anything's metric model, whose scale
  was 1.3–4× off on PointOdyssey. Camera translation and 3D tracks inherit that scale. Ratios and relative geometry
  are reliable; absolute metres are not.
- Mask names: only objects in the `--vocab` list get a name; most surfaces and background parts are `unknown`.
- Fast-moving thin objects are tracked less reliably (CoTracker: about 9 px median error on PointOdyssey's animated
  characters vs 0.6 px on static points).
- Tested on clips of up to 180 frames. Longer videos are processed in segments, but runs on videos of several
  thousand frames haven't been done yet.
