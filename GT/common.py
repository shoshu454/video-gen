# Shared loaders and geometry helpers for the demo QA pipeline.
# Every loader returns a plain dict:
#   n_frames, fps, H, W, K (3x3, pixels)
#   P_cam (T, N, 3)  3D track positions in the camera frame (OpenCV: x right, y down, z fwd)
#   uv    (T, N, 2)  2D track positions in pixels (projection of P_cam)
#   vis   (T, N)     per-frame visibility from the dataset
#   inst  (N,)       object / surface id per track (DR instances; PO: -1 unknown)
#   image(i) -> HxWx3 uint8, depth(i) -> HxW float32 z-depth in meters
# See CONVENTIONS.md for the evidence behind each conversion.
import gzip, io, json, os
import numpy as np
import torch
from PIL import Image
from blk import DR, PO, names, read

OUT_ROOT = '/shared/rsaas/joeyw3/pvideo_demo'
CACHE = f'{OUT_ROOT}/cache'
DR_ANN = '/shared/rsaas/joeyw3/data/dynamic_replica/frame_annotations_train.jgz'
PY3D_TO_CV = np.diag([-1.0, -1.0, 1.0])  # PyTorch3D camera (x left, y up) -> OpenCV


def dr_sequences():
    return sorted(s for s in os.listdir(DR) if os.path.exists(f'{DR}/{s}/.complete'))


def po_scenes():
    out = []
    for split in sorted(os.listdir(PO)):
        for s in sorted(os.listdir(f'{PO}/{split}')):
            if os.path.exists(f'{PO}/{split}/{s}/.complete') and os.path.exists(f'{PO}/{split}/{s}/anno.npz'):
                out.append(f'{split}/{s}')
    return out


_dr_ann = None


def dr_annotations(seq):
    global _dr_ann
    if _dr_ann is None:
        _dr_ann = {}
        for a in json.load(gzip.open(DR_ANN)):
            _dr_ann.setdefault(a['image']['path'].split('/')[0], {})[a['frame_number']] = a
    return _dr_ann[seq]


def load_dr(seq):
    d = f'{DR}/{seq}'
    ann = dr_annotations(seq)
    T = len(names(d, 'trajectories'))
    a0 = ann[0]
    H, W = a0['image']['size']
    s = min(H, W) / 2
    f = a0['viewpoint']['focal_length'][0] * s
    K = np.array([[f, 0, W / 2 - a0['viewpoint']['principal_point'][0] * s],
                  [0, f, H / 2 - a0['viewpoint']['principal_point'][1] * s], [0, 0, 1]])
    cache = f'{CACHE}/dr_{seq}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        P_cam, uv, vis, inst = z['P_cam'], z['uv'], z['vis'], z['inst']
    else:
        P_cam, uv, vis = [], [], []
        for i in range(T):
            t = torch.load(io.BytesIO(read(d, 'trajectories', f'{i:06d}.pth')), weights_only=False)
            vp = ann[i]['viewpoint']
            R, Tt = np.array(vp['R']), np.array(vp['T'])
            Xc = (t['traj_3d_world'].numpy().astype(np.float64) @ R + Tt) @ PY3D_TO_CV  # world -> cam (row vectors)
            P_cam.append(Xc.astype(np.float32))
            uv.append(t['traj_2d'][:, :2].numpy())
            vis.append(t['verts_inds_vis'].numpy())
            if i == 0:
                inst = t['instances'].numpy().astype(np.int32)
        P_cam, uv, vis = np.stack(P_cam), np.stack(uv), np.stack(vis)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache + '.tmp.npz', P_cam=P_cam, uv=uv, vis=vis, inst=inst)
        os.replace(cache + '.tmp.npz', cache)
    img_names = [n for n in names(d, 'images') if n.endswith('.png')]
    dep_names = names(d, 'depths')

    def image(i):
        return np.array(Image.open(io.BytesIO(read(d, 'images', img_names[i]))).convert('RGB'))

    def depth(i):
        raw = np.array(Image.open(io.BytesIO(read(d, 'depths', dep_names[i]))))
        return np.frombuffer(raw.astype(np.uint16).tobytes(), dtype=np.float16).astype(np.float32).reshape(H, W)

    fps = 1.0 / np.median(np.diff([ann[i]['frame_timestamp'] for i in range(T)]))
    return dict(dataset='dynamic_replica', video_id=seq, n_frames=T, fps=float(round(fps, 3)), H=H, W=W, K=K,
                P_cam=P_cam, uv=uv, vis=vis, inst=inst, image=image, depth=depth)


PO_FPS = 24.0  # frame rate of the dataset's own preview mp4s (849 frames / 35.38 s); see CONVENTIONS.md


def load_po(scene):
    d = f'{PO}/{scene}'
    a = np.load(f'{d}/anno.npz')
    T3, E, K = a['trajs_3d'], a['extrinsics'], a['intrinsics']
    P_cam = np.einsum('tij,tnj->tni', E[:, :3, :3], T3) + E[:, None, :3, 3]  # extrinsics are world -> cam (OpenCV)
    vis = a['visibs'] & a['valids']
    first = np.array(Image.open(io.BytesIO(read(d, 'rgbs', 'rgb_00000.jpg'))))
    H, W = first.shape[:2]

    def image(i):
        return np.array(Image.open(io.BytesIO(read(d, 'rgbs', f'rgb_{i:05d}.jpg'))).convert('RGB'))

    def depth(i):
        raw = np.array(Image.open(io.BytesIO(read(d, 'depths', f'depth_{i:05d}.png')))).astype(np.float32)
        return raw * (1000.0 / 65535.0)

    def mask(i):
        return np.array(Image.open(io.BytesIO(read(d, 'masks', f'mask_{i:05d}.png'))))

    return dict(dataset='pointodyssey', video_id=scene, n_frames=len(T3), fps=PO_FPS, H=H, W=W, K=K[0].astype(np.float64),
                P_cam=P_cam.astype(np.float32), uv=a['trajs_2d'].astype(np.float32), vis=vis, valid=a['valids'],
                inst=np.full(T3.shape[1], -1, np.int32), image=image, depth=depth, mask=mask)


def euclid(P):
    """Euclidean distance to the camera center for camera-frame points (..., 3)."""
    return np.linalg.norm(P, axis=-1)


def patch_rel_std(depth, u, v, half=2):
    """Relative std (std/mean) of z-depth in a (2*half+1)^2 patch; inf if out of bounds or empty."""
    H, W = depth.shape
    u, v = int(round(u)), int(round(v))
    if u - half < 0 or v - half < 0 or u + half >= W or v + half >= H:
        return np.inf
    p = depth[v - half:v + half + 1, u - half:u + half + 1]
    p = p[p > 0]
    if len(p) < (2 * half + 1) ** 2:
        return np.inf
    return float(p.std() / p.mean())


def in_bounds(uv, W, H, pad=0):
    return (uv[..., 0] >= pad) & (uv[..., 0] < W - pad) & (uv[..., 1] >= pad) & (uv[..., 1] < H - pad)


def bake_indices(start, end, forced, n=32):
    """n frame indices spanning [start, end] (inclusive) roughly uniformly, containing every forced
    index. Each forced index replaces the nearest not-yet-forced uniform sample."""
    idx = [int(round(x)) for x in np.linspace(start, end, n)]
    locked = set()
    for f in sorted(set(forced)):
        if f in idx:
            locked.add(idx.index(f))
            continue
        free = [k for k in range(n) if k not in locked]
        k = min(free, key=lambda k: abs(idx[k] - f))
        idx[k] = f
        locked.add(k)
    idx = sorted(idx)
    return idx
