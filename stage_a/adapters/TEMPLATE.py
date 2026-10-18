# Adapter TEMPLATE: copy to adapters/<your_dataset>.py and fill in the read_* functions.
#   python adapters/<your_dataset>.py <dataset_root> <out_dir>
# Each read_* function gets one item of list_videos() and returns the modality in the units / conventions below, or
# None if the dataset does not have it (stage_a.py estimates missing modalities). Do the unit and convention
# conversions HERE (see README_FORMAT.md, "Pitfalls"); the shared code at the bottom only checks shapes and writes.
# Run `python check_inputs.py <out_dir>` afterwards: its static-point reprojection check catches most convention bugs.
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import stdfmt as F

DATASET = 'MY_DATASET'  # TODO: dataset name written to meta.json


def list_videos(root):
    """TODO: return a list of (video_name, source_path) for every video in the dataset.
    video_name becomes the output folder name: unique, no '/', e.g. f'{scene}_{camera}'."""
    raise NotImplementedError


def read_fps(src):
    """TODO: frames per second of the video (float)."""
    raise NotImplementedError


def read_rgb(src):
    """TODO: (T, H, W, 3) uint8 RGB (not BGR: convert cv2.imread output with [..., ::-1]), undistorted pinhole.
    Alternatively return the path of an .mp4 file, which is linked as is (its frames define T, H, W)."""
    raise NotImplementedError


def read_depth(src):
    """(T, H, W) float metres, z-depth along the optical axis (NOT distance along the ray), nan or 0 = invalid.
    Typical conversions: uint16 PNG in mm -> / 1000; ray distance r -> z = r / ||K^-1 [u, v, 1]||; disparity ->
    z = f * baseline / disparity. Must be at the RGB resolution (resize with nearest-neighbour if needed)."""
    return None


def read_masks(src):
    """(T, H, W) int instance ids, 0 = background; the same object keeps the same id in every frame.
    Returns (masks, names) where names is {id: 'person', ...} or None. Semantic-only (class) maps are not instance ids."""
    return None


def read_intrinsics(src):
    """(3, 3) K in pixels of the RGB frames: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. If the dataset resized or cropped
    the images, scale / shift K to match. Normalised intrinsics (0..1) must be multiplied by W / H."""
    return None


def read_extrinsics(src):
    """(T, 4, 4) camera-to-world (c2w), OpenCV camera axes (x right, y down, z forward), translation in metres.
    World-to-camera poses: np.linalg.inv. OpenGL / Blender cameras (y up, z backward): c2w @ diag(1, -1, -1, 1)."""
    return None


def read_tracks(src):
    """dict(tracks_2d=(T, N, 2) xy pixels, visibility=(T, N) bool, query_frame=(N,) int or None,
    tracks_3d=(T, N, 3) world metres (same world as the extrinsics) or None), or None."""
    return None


# ---------------------------------------------------------------- shared code: no changes needed below
def convert(root, out_dir, only=None):
    vids = list_videos(root)
    for name, src in vids:
        if only and name not in only:
            continue
        rgb = read_rgb(src)
        mp4 = rgb if isinstance(rgb, str) else None
        m = read_masks(src)
        masks, names = (m if isinstance(m, tuple) else (m, None)) if m is not None else (None, None)
        tr = read_tracks(src) or {}
        F.write_video_folder(out_dir, name, dataset=DATASET, source_path=src, fps=read_fps(src),
                             rgb=None if mp4 else rgb, rgb_mp4=mp4, depth_m=read_depth(src), masks=masks, names=names,
                             intrinsics=read_intrinsics(src), extrinsics_c2w=read_extrinsics(src),
                             tracks_2d=tr.get('tracks_2d'), visibility=tr.get('visibility'),
                             query_frame=tr.get('query_frame'), tracks_3d=tr.get('tracks_3d'))
        print('wrote', os.path.join(out_dir, name), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('out_dir')
    ap.add_argument('--only', default='', help='comma list of video names')
    a = ap.parse_args()
    convert(a.root, a.out_dir, set(a.only.split(',')) if a.only else None)
