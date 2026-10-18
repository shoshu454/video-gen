# Standard video geometry format

This is the input **and** output format of `stage_a.py`. To bring a new dataset in, write an *adapter* that converts
it to this format: copy `adapters/TEMPLATE.py` and fill in its `read_*` functions. Then run
`python check_inputs.py <dir>` and fix every error and warning before running `stage_a.py`. Every rule is in this
file; you shouldn't need to ask anyone.

## Layout

```
<dir>/
  <video_name>/                  one folder per video; name: letters, digits, '_', '-', '.' (no '/')
    video.mp4                    the RGB frames as an H.264 .mp4 ...
    rgb/00000.png 00001.png ...  ... OR as PNG frames (exactly one of the two)
    meta.json                    REQUIRED
    depth/00000.png ...          optional  16-bit PNG, z-depth in millimetres, 0 = invalid
    masks/00000.png ...          optional  16-bit PNG, instance ids, 0 = background
    masks/names.json             optional  {"<instance id>": "<object name>"}
    tracks/tracks_2d.npy         optional  (T, N, 2) float32, x, y in pixels
    tracks/visibility.npy        required with tracks_2d.npy: (T, N) bool
    tracks/query_frame.npy       optional  (N,) int64, the frame each track was started from
    tracks/tracks_3d.npy         optional  (T, N, 3) float32, world coordinates, metres
    camera/intrinsics.npy        optional  (3, 3) float32 K, pixels
    camera/extrinsics.npy        optional  (T, 4, 4) float32 camera-to-world, OpenCV axes, metres
```

T is the number of RGB frames and N the number of tracks. File names are zero-padded to 5 digits, start at `00000`
and have no gaps. Frame `i` of every modality is RGB frame `i`.

**All or nothing per modality:** a modality is either present for all T frames or absent (no folder / no file).
If your dataset has depth for only some frames, leave depth out entirely (stage_a will estimate it), or fill the
missing frames with 0 (invalid). Any modality may be missing for any video.

## meta.json (required)

```json
{
  "dataset": "HOI4D",
  "source_path": "/data/HOI4D/ZY2021.../align_rgb",
  "fps": 15,
  "num_frames": 300,
  "resolution": [1920, 1080]
}
```

| key | meaning |
|---|---|
| `dataset` | dataset name (free text) |
| `source_path` | where this video came from in the original dataset (for traceability) |
| `fps` | frames per second of the *stored* frames (if you subsample every 2nd frame of a 30 fps video, write 15) |
| `num_frames` | T: must equal the number of RGB frames |
| `resolution` | `[W, H]`, width first, of the RGB frames; every other modality has the same size |

You may add more keys. stage_a copies them to its output.

## Modalities

**RGB.** 8-bit RGB (not BGR). The images must be undistorted pinhole images (see Pitfalls). `video.mp4` must decode
to exactly `num_frames` frames: use a constant frame rate and no B-frame edit lists. If unsure, write `rgb/` PNGs.

**depth/** One 16-bit single-channel PNG per frame, at the RGB resolution. Each value is **z-depth along the optical
axis in millimetres**, as an unsigned integer. `0` means invalid or unknown (sky, sensor holes, beyond range). The
largest value you can store is 65535 mm = 65.5 m. Store anything farther as 0.
Conversion: `png = np.round(z_metres * 1000).astype(np.uint16)` with invalid or too-far pixels set to 0.

**masks/** One 16-bit single-channel PNG per frame, at the RGB resolution. Each value is an **instance id**:
`0` = background or unlabelled, `1..65535` = object instances. **The same physical object must keep the same id in
every frame**, so ids that are only unique within one frame don't qualify. Semantic class maps (all cars = 3) are not
instance masks. 8-bit PNGs are accepted too. `names.json` (optional) maps ids (as strings) to names:
`{"1": "person", "2": "mug"}`.

**tracks/** Point tracks over the whole video.
- `tracks_2d.npy` (T, N, 2) float32: pixel x (column), y (row) of track n at frame t. Use the pixel-centre
  convention: pixel (0, 0) covers [−0.5, 0.5]². Every frame needs a finite value. When a point is not visible, store
  its estimated position or repeat the last known one.
- `visibility.npy` (T, N) bool: True where the point is visible, i.e. not occluded and inside the image.
- `query_frame.npy` (N,) int64 (optional): the frame where each track was initialised.
- `tracks_3d.npy` (T, N, 3) float32 (optional): the point's position in the **same world frame as
  `extrinsics.npy`**, in metres. Use `nan` where it is unknown.

**camera/intrinsics.npy** (3, 3) float32, one K for the whole video, in pixels of the stored frames:
```
[[fx, 0,  cx],
 [0,  fy, cy],
 [0,  0,  1 ]]
```
A pixel (u, v) at depth z maps to the camera-frame point `z * K^-1 [u, v, 1]`. If the focal length changes during
the video (zoom), the format can't represent it: split the video, or leave out intrinsics and extrinsics.

**camera/extrinsics.npy** (T, 4, 4) float32, **camera-to-world** (c2w) per frame, with **OpenCV camera axes**
(x right, y down, z forward, i.e. the viewing direction) and translation in **metres**. A camera-frame point P maps to
the world point `X = c2w[t] @ [P, 1]`. The world frame is arbitrary; only its scale must be metric. The rotation part
must be a proper rotation (orthonormal, det = +1). The last row is `[0, 0, 0, 1]`.

## Pitfalls (check every one)

1. **Depth units.** The format stores millimetres. Common sources: metres as float (×1000), millimetres as uint16
   (as is), PointOdyssey `raw * 1000 / 65535` m, ScanNet / TUM uint16 `/ 1000` or `/ 5000` m, KITTI depth PNG `/ 256` m.
   `check_inputs.py` prints `depth_median_m`. Indoor scenes are typically 1–5 m and driving scenes 5–30 m.
2. **z-depth vs ray distance.** z is measured along the optical axis. Some renderers (and "range" or "distance" maps)
   store the Euclidean distance r to the camera centre instead. Convert with
   `z = r / ||K^-1 [u, v, 1]||`. Otherwise walls look curved and `check_inputs.py` reports a large depth reprojection error.
3. **Disparity vs depth.** Stereo or relative-depth models output disparity or inverse depth. Convert to metric z
   (`z = f * baseline / disparity`), or leave depth out.
4. **c2w vs w2c.** Many datasets (COLMAP, PointOdyssey `extrinsics`, KITTI `Tr`) store **world-to-camera**. Invert:
   `c2w = np.linalg.inv(w2c)`. `check_inputs.py` runs the static-point check both ways and warns
   "extrinsics look like WORLD-TO-CAMERA".
5. **OpenGL / Blender vs OpenCV axes.** OpenGL, Blender, NeRF `transforms.json` and ARKit cameras look down −z with
   y up. Convert: `c2w_cv = c2w_gl @ np.diag([1, -1, -1, 1])`. A left-handed source (Unity) gives det(R) = −1, which
   is an error.
6. **Intrinsics of resized or cropped images.** K must match the stored frames. When resizing by s, multiply fx, fy,
   cx, cy by s (exactly: `cx' = (cx + 0.5) * s - 0.5`). When cropping by (x0, y0), subtract them from cx, cy. Normalised
   intrinsics (fx in units of image width) must be multiplied by W (and H for fy, cy).
7. **Fisheye / distortion.** The format is pinhole only. Undistort RGB, depth (nearest-neighbour), masks
   (nearest-neighbour) and tracks with the same mapping (e.g. `cv2.fisheye.initUndistortRectifyMap` or
   `cv2.undistortPoints`), and store the new K. Don't store distorted images with a pinhole K.
8. **Frame alignment.** Frame i of depth, masks, tracks and poses must be the moment of RGB frame i. Watch for
   depth or pose streams at a different rate (match by timestamp and drop unmatched frames), off-by-one from 1-based
   file names, and dropped frames in the video. If you subsample frames, subsample all modalities identically and
   update `fps`. If decoding the source .mp4 gives a different frame count than the dataset says, write PNG frames.
9. **Depth / mask resolution.** Depth sensors often have a lower resolution (e.g. 640×480 depth with 1920×1080 RGB).
   Register depth to the RGB camera (the dataset usually provides aligned depth) and resize with nearest-neighbour,
   not bilinear: bilinear invents depths between foreground and background.
10. **Mask ids across frames.** Per-frame detector or segmentation outputs are not consistent instance ids.
    Leave masks out rather than storing per-frame ids.
11. **Track coordinates.** (x, y) order, not (row, col). Normalised (0..1) coordinates must be multiplied by (W, H).
12. **BGR.** `cv2.imread` returns BGR. Convert with `[..., ::-1]` before writing.

## Validating

```bash
python check_inputs.py <dir>            # all checks; --quick reads only the first and last depth / mask PNG
```

For each video it prints `[OK]`, `[WARN]` or `[ERROR]` plus measurements:
- `depth_median_m`: should look like the scene's depth.
- `reproj_depth_relerr`: background pixels of frame t are unprojected with depth, moved to frame t+k with the
  extrinsics and compared with the depth of frame t+k. Should be < 0.05. `_if_w2c` is the same check with the poses
  inverted: if *that* one is clearly smaller, your poses are world-to-camera.
- `track_reproj_px`: static tracks moved from t to t+k the same way, compared with `tracks_2d`. Should be a few pixels.
  Static means unmoved in `tracks_3d` if given, otherwise the best 25% (`track_reproj_px_p25`).
- `tracks3d_reproj_px`: `tracks_3d` projected with extrinsics and intrinsics vs `tracks_2d`. Should be < 2 px.

These checks need depth, intrinsics and extrinsics together. With less GT, you can only check them by looking.

## What stage_a.py adds to the output

The output uses the same format with every modality present, plus:
- `meta.json` gets `"sources": {"depth": "gt" | "<model>", "masks": ..., "tracks": ..., "intrinsics": ...,
  "extrinsics": ...}` and `"scale_alignment": {...}` (the scale factors applied when GT and estimates are mixed).
- `DONE`: an empty file written last.
- Estimated tracks always include `query_frame.npy` and `tracks_3d.npy`. With GT `tracks_2d` but no `tracks_3d`,
  stage_a adds `tracks_3d` by lifting (`nan` for tracks it can't lift).
