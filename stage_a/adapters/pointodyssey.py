# PointOdyssey adapter (worked example of TEMPLATE.py) for the packed scenes of ../GT/pack_po.py.
#   python adapters/pointodyssey.py <out_dir> --scenes test/ani2,sample/r4_new_f [--start 0 --frames 120]
#       [--drop tracks,masks] [--max_tracks 4000] [--name_suffix _part]
# PO conventions -> standard format:
#   depth  16-bit PNG, z = raw * 1000 / 65535 m           -> metres (written as mm)
#   masks  8-bit palette PNG of instance ids              -> ids as is (0 = background)
#   extrinsics  world-to-camera, OpenCV                   -> inverted to camera-to-world
#   trajs_2d / visibs & valids / trajs_3d (world)         -> tracks_2d / visibility / tracks_3d
#   intrinsics  per frame (T, 3, 3)                       -> one K; scenes whose K changes (zoom) are rejected
# --drop removes modalities (to make partial-GT test inputs).
import argparse, io, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), 'GT')]
import numpy as np
from PIL import Image
import stdfmt as F

PO_FPS = 24.0  # see GT/common.py


def convert(scene, out_dir, start=0, frames=0, drop=(), max_tracks=0, suffix='', seed=0):
    from blk import PO, read
    d = f'{PO}/{scene}'
    a = np.load(f'{d}/anno.npz')
    T_all = len(a['extrinsics'])
    end = T_all if not frames else min(T_all, start + frames)
    sl = slice(start, end)
    Ks = a['intrinsics'][sl]
    if np.abs(Ks - Ks[0]).max() > 1e-3:
        raise ValueError(f'{scene}: intrinsics change over frames {start}..{end - 1} (zoom); not representable')
    img = lambda kind, f: np.array(Image.open(io.BytesIO(read(d, kind, f))))
    rgb = np.stack([img('rgbs', f'rgb_{i:05d}.jpg')[..., :3] for i in range(start, end)])
    depth = None if 'depth' in drop else np.stack(
        [img('depths', f'depth_{i:05d}.png').astype(np.float32) * (1000.0 / 65535.0) for i in range(start, end)])
    masks = None if 'masks' in drop else np.stack([img('masks', f'mask_{i:05d}.png') for i in range(start, end)])
    c2w = None if 'extrinsics' in drop else np.linalg.inv(a['extrinsics'][sl].astype(np.float64))
    K = None if 'intrinsics' in drop else Ks[0]
    tr = {}
    if 'tracks' not in drop:
        uv, vis, X = a['trajs_2d'][sl], (a['visibs'] & a['valids'])[sl], a['trajs_3d'][sl]
        j = np.nonzero(vis.any(0))[0]  # tracks seen in this window
        if max_tracks and len(j) > max_tracks:
            j = np.sort(np.random.default_rng(seed).choice(j, max_tracks, replace=False))
        tr = dict(tracks_2d=uv[:, j], visibility=vis[:, j], query_frame=np.argmax(vis[:, j], 0),
                  tracks_3d=None if 'tracks_3d' in drop else X[:, j])
    name = scene.replace('/', '_') + suffix
    return F.write_video_folder(out_dir, name, dataset='PointOdyssey', source_path=f'{d} [{start}:{end}]', fps=PO_FPS,
                                rgb=rgb, depth_m=depth, masks=masks, intrinsics=K, extrinsics_c2w=c2w,
                                extra_meta=dict(source_frames=[start, end]), **tr)


if __name__ == '__main__':
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE (vision-c22)
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--scenes', required=True, help='comma list of <split>/<scene>')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0, help='0 = all')
    ap.add_argument('--drop', default='', help='comma list of depth,masks,intrinsics,extrinsics,tracks,tracks_3d')
    ap.add_argument('--max_tracks', type=int, default=0)
    ap.add_argument('--name_suffix', default='')
    a = ap.parse_args()
    for s in a.scenes.split(','):
        print('wrote', convert(s, a.out_dir, a.start, a.frames, set(filter(None, a.drop.split(','))), a.max_tracks,
                               a.name_suffix), flush=True)
