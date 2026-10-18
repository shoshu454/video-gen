# Pseudo-GT builder: run the specialist models on a video and write the result in the exact on-disk formats the
# pipeline reads, so the unmodified walkthrough / prod code runs on it with only its data roots redirected.
#   PO format (T1, T3, T4, T5):  <PO_ROOT>/pseudo/<name>/{anno.npz, rgbs|depths|masks.blk + .idx.json, .complete}
#       anno.npz: trajs_2d, trajs_3d (world), visibs, valids, intrinsics (T,3,3), extrinsics (T,4,4 world->cam)
#       depths: 16-bit PNG, z = raw * 1000 / 65535 m (PointOdyssey encoding); masks: 8-bit instance ids
#   KITTI-raw format (T6):        <KITTI_ROOT>/<date>/{calib_*.txt}, <date>/<drive>_sync/{image_02.blk, velodyne.blk,
#       oxts/data/*.txt, image_02_timestamps.txt, tracklet_labels.xml, .blk_complete}; <date> = drive[:10]
# Specialist outputs are cached in <RAW>/<name>.npz so the fusion can be redone without the GPU.
#   python build.py clip T3_001                 # one clip from clips.json (PO scene or KITTI drive, RGB only)
#   python build.py video in.mp4 --name foo [--kitti]   # any video
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE: THP compaction stalls make big allocations take minutes on vision-c22
import argparse, io, json, os, shutil, sys, time, zlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
ROOT = '/shared/rsaas/joeyw3/pvideo_pseudo'
RAW = f'{ROOT}/raw'
PO_ROOT = f'{ROOT}/data/po'  # scenes are 'pseudo/<name>' under it
KITTI_ROOT = '/tmp/claude-1267708/pgt_kitti'  # local disk: the KITTI layout has one small oxts file per frame
CLIPS = f'{ROOT}/clips.json'
SCRATCH = os.environ.get('PGT_SCRATCH', '/tmp/claude-1267708/pgt_scratch')  # same local-disk scratch as specialists.py
PO_FPS = 24.0  # common.load_po hard-codes 24 fps, so PO-format scenes are resampled to it


# ---------------------------------------------------------------- frame sources (RGB only)
def frames_po(scene, start, end):
    from blk import PO, read
    d = f'{PO}/{scene}'
    return np.stack([np.array(Image.open(io.BytesIO(read(d, 'rgbs', f'rgb_{i:05d}.jpg'))).convert('RGB'))
                     for i in range(start, end + 1)]), PO_FPS


def frames_kitti(drive, start, end):
    from blk import names, read
    dd = f'/shared/rsaas/joeyw3/data/kitti_raw/{drive[:10]}/{drive}_sync'
    ts = [l.strip() for l in open(f'{dd}/image_02_timestamps.txt')]
    sec = np.array([int(t[11:13]) * 3600 + int(t[14:16]) * 60 + float(t[17:]) for t in ts])
    # by file name, not list position: the scale-up fetch packs only the clip windows of some drives
    fr = np.stack([np.array(Image.open(io.BytesIO(read(dd, 'image_02', f'{i:010d}.png'))).convert('RGB'))
                   for i in range(start, end + 1)])
    return fr, float(round(1 / np.median(np.diff(sec)), 3))


def frames_video(path, fps_out=None, max_long=1280):
    import imageio.v3 as iio
    meta = iio.immeta(path, plugin='pyav')
    fps = float(meta.get('fps', 30))
    fr = np.stack([f for f in iio.imiter(path, plugin='pyav')])
    if fps_out and abs(fps_out - fps) > 1e-3:  # nearest-frame resampling to the target rate
        n = int(np.floor((len(fr) - 1) / fps * fps_out)) + 1
        fr = fr[np.minimum(np.round(np.arange(n) / fps_out * fps).astype(int), len(fr) - 1)]
        fps = fps_out
    H, W = fr.shape[1:3]
    if max(H, W) > max_long:
        import cv2
        s = max_long / max(H, W)
        sz = (int(round(W * s / 2)) * 2, int(round(H * s / 2)) * 2)
        fr = np.stack([cv2.resize(f, sz, interpolation=cv2.INTER_AREA) for f in fr])
    return fr, fps


# ---------------------------------------------------------------- run specialists (GPU)
def gpu_peak(reset=True):
    """Peak GPU memory (GiB) allocated by this process since the last reset."""
    import torch
    g = torch.cuda.max_memory_allocated() / 2 ** 30
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return g


def child_env():
    """Environment of a GPU child process: on PGT_CHILD_GPU if set (2-GPU jobs put SAM2 / the detector on the
    second GPU: SAM2 (~25 GiB) + MonST3R (~19 GiB) do not fit on one 44 GiB A40), else on the parent's GPU."""
    env = dict(os.environ)
    if os.environ.get('PGT_CHILD_GPU'):
        env['CUDA_VISIBLE_DEVICES'] = os.environ['PGT_CHILD_GPU']
    return env


def start_sam2(frames, fps):
    """SAM2 instance masks in a child process, so it runs alongside VDA / MonST3R / CoTracker
    (a thread would race on MonST3R's os.chdir and on the sys.modules swaps between the two sam2 packages)."""
    import subprocess, tempfile
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='sam2job_', dir=SCRATCH)
    np.save(f'{tmp}/frames.npy', frames)
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), 'sam2', tmp, '--fps', str(fps)], env=child_env())
    return proc, tmp


def finish_sam2(job):
    proc, tmp = job
    if proc.wait() != 0:
        raise RuntimeError(f'SAM2 child failed (exit {proc.returncode})')
    labels = np.load(f'{tmp}/labels.npy')
    shutil.rmtree(tmp, ignore_errors=True)
    return labels


def kill_child(job):
    """A failed parent must not leave its GPU child running (it would hold its memory and block the retry)."""
    if job is not None and job[0].poll() is None:
        job[0].kill()
        job[0].wait()


def run_specialists(frames, fps, kitti=False, probes=None, boxes=None, parallel=False, det_job=None):
    """boxes: Grounding DINO boxes (KITTI), or None with det_job = a running `build.py detect` child (parallel)."""
    sam2_job = start_sam2(frames, fps) if parallel and not kitti else None
    try:
        return _run_specialists(frames, fps, kitti, probes, boxes, sam2_job, det_job)
    except BaseException:
        kill_child(sam2_job)
        kill_child(det_job)
        raise


def _run_specialists(frames, fps, kitti, probes, boxes, sam2_job, det_job):
    import specialists as S
    out = {}
    t_all = time.time()
    t = time.time()
    out['depth'] = S.run_vda(frames, fps).astype(np.float16)
    print(f'VDA {time.time() - t:.0f} s (peak {gpu_peak():.1f} GiB)', flush=True)
    t = time.time()
    m = S.run_monst3r(frames, max_frames=50)  # an initialisation for bundle_adjust (and the metric scale)
    out.update({f'mon_{k}': v for k, v in m.items()})
    print(f'MonST3R {time.time() - t:.0f} s ({len(m["sel"])} frames, peak {gpu_peak():.1f} GiB)', flush=True)
    if kitti:
        t = time.time()
        if det_job is not None:
            boxes = finish_detect(*det_job)
            print(f'detector (parallel) waited {time.time() - t:.0f} s after MonST3R', flush=True)
            t = time.time()
        det = S.run_detect_objects(frames, boxes)
        out['det_frame'] = np.array([f for f, items in enumerate(det) for _ in items], np.int32)
        out['det_type'] = np.array([ty for items in det for ty, _ in items])
        out['det_mask'] = np.packbits(np.stack([m for items in det for _, m in items]) if out['det_frame'].size
                                      else np.zeros((0,) + frames.shape[1:3], bool), axis=-1)
        print(f'detector {time.time() - t:.0f} s ({len(out["det_frame"])} masks)', flush=True)
        t = time.time()
        c = S.run_cotracker(frames, fps, every_sec=0.5)  # correspondences for metric visual odometry
        out.update({f'ct_{k}': v for k, v in c.items()})
        print(f'CoTracker {time.time() - t:.0f} s ({c["uv"].shape[1]} tracks)', flush=True)
    else:
        t = time.time()
        c = S.run_cotracker(frames, fps, extra=probes)
        out.update({f'ct_{k}': v for k, v in c.items()})
        print(f'CoTracker {time.time() - t:.0f} s ({c["uv"].shape[1]} tracks, peak {gpu_peak():.1f} GiB)', flush=True)
        t = time.time()
        if sam2_job:
            out['labels'] = finish_sam2(sam2_job)
            print(f'SAM2 (parallel) waited {time.time() - t:.0f} s after CoTracker ({out["labels"].max()} instances)', flush=True)
        else:
            out['labels'] = S.run_sam2_instances(frames, fps)
            print(f'SAM2 {time.time() - t:.0f} s ({out["labels"].max()} instances, peak {gpu_peak():.1f} GiB)', flush=True)
    print(f'GPU stages {time.time() - t_all:.0f} s', flush=True)
    return out


# ---------------------------------------------------------------- camera fusion
def interp_poses(sel, c2w, T):
    """cam-to-world (n, 4, 4) at frame indices sel -> (T, 4, 4): slerp rotations, lerp translations."""
    from scipy.spatial.transform import Rotation, Slerp
    t = np.arange(T)
    rot = Slerp(sel, Rotation.from_matrix(c2w[:, :3, :3]))(np.clip(t, sel[0], sel[-1])).as_matrix()
    out = np.tile(np.eye(4), (T, 1, 1))
    out[:, :3, :3] = rot
    for k in range(3):
        out[:, k, 3] = np.interp(t, sel, c2w[:, k, 3])
    return out


def metric_scale(raw, depth):
    """MonST3R's reconstruction has an arbitrary scale: the median ratio of VDA metric depth to MonST3R depth over
    confident, static pixels of the MonST3R frames."""
    import cv2
    s, x0, y0, w, h = raw['mon_crop']
    x0, y0, w, h = int(x0), int(y0), int(w), int(h)
    H1, W1 = depth.shape[1:]
    rs = []
    for k, i in enumerate(raw['mon_sel']):
        dv = cv2.resize(depth[i].astype(np.float32), (int(round(W1 * s)), int(round(H1 * s))), interpolation=cv2.INTER_NEAREST)
        dv = dv[y0:y0 + h, x0:x0 + w]
        dm, cf, dy = raw['mon_depth'][k], raw['mon_conf'][k], raw['mon_dyn'][k]
        ok = (dm > 1e-4) & (dv > 0.1) & (dv < 80) & ~dy & (cf >= np.percentile(cf, 50))
        if ok.sum() > 100:
            rs.append(np.median(dv[ok] / dm[ok]))
    return float(np.median(rs))


def _rotvec_to_R(w):
    """(..., 3) axis-angle -> (..., 3, 3) rotation (Rodrigues), torch."""
    import torch
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    Kx = Kx - Kx.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, dtype=w.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def bundle_adjust(uv, vis, depth, c2w, K, n_max=3000, iters=400, huber=2.0, prior=0.05):
    """Refine camera poses with the dense 2D tracks: per-frame pose corrections (frame 0 fixed) and one inverse
    depth per static-candidate track (along its first visible ray), Huber reprojection loss + a log-depth prior
    towards the metric VDA depth (fixes the scale). Candidates: tracks whose static-hypothesis residual is in the
    lower half; a second round re-selects them with the refined cameras. Returns refined cam-to-world (T, 4, 4) and K (shared focal refined)."""
    import torch
    T, N = vis.shape
    Ki = np.linalg.inv(K)
    for rnd in range(2):
        Xo = np.full((T, N, 3), np.nan, np.float32)
        for t in range(T):
            z = sample_depth(depth, t, uv[t])
            ok = vis[t] & np.isfinite(z) & (z > 0.05)
            Pc = (np.c_[uv[t], np.ones(N)] @ Ki.T) * z[:, None]
            Xo[t, ok] = (Pc @ c2w[t, :3, :3].T + c2w[t, :3, 3])[ok]
        okv = np.isfinite(Xo[..., 0])
        res = static_residual(Xo, okv, uv, c2w, K)
        nvis = okv.sum(0)
        cand = np.nonzero((nvis >= 10) & np.isfinite(res))[0]
        if len(cand) < 50:
            return c2w, K
        cand = cand[res[cand] <= np.percentile(res[cand], 50)]
        cand = np.random.default_rng(rnd).permutation(cand)[:n_max]
        ref = np.argmax(okv[:, cand], 0)
        uref = uv[ref, cand]
        zref = np.array([sample_depth(depth, int(r), u[None])[0] for r, u in zip(ref, uref)], np.float64)
        good = np.isfinite(zref) & (zref > 0.05)
        cand, ref, uref, zref = cand[good], ref[good], uref[good], zref[good]
        to = lambda a: torch.tensor(a, dtype=torch.float64)
        R0, C0 = to(c2w[:, :3, :3]), to(c2w[:, :3, 3])
        Kt, Kit = to(K), to(Ki)
        ray = to(np.c_[uref, np.ones(len(uref))]) @ Kit.T  # (n, 3) camera ray of the reference pixel
        obs_t, obs_n = np.nonzero(vis[:, cand])
        obs_uv = to(uv[obs_t, cand[obs_n]])
        obs_t, obs_n = torch.tensor(obs_t), torch.tensor(obs_n)
        dw = torch.zeros(T, 3, dtype=torch.float64, requires_grad=True)
        dc = torch.zeros(T, 3, dtype=torch.float64, requires_grad=True)
        logz = to(np.log(zref)).clone().requires_grad_(True)
        logz0 = to(np.log(zref))
        refT = torch.tensor(ref)
        logf = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # shared focal correction
        opt = torch.optim.Adam([dw, dc, logz, logf], lr=2e-3)
        mask = torch.ones(T, 1, dtype=torch.float64)
        mask[0] = 0  # gauge: frame 0 fixed
        for it in range(iters):
            opt.zero_grad()
            R = _rotvec_to_R(dw * mask) @ R0
            C = C0 + dc * mask
            Kf = Kt.clone()
            Kf[0, 0], Kf[1, 1] = Kt[0, 0] * torch.exp(logf[0]), Kt[1, 1] * torch.exp(logf[0])
            rayf = ray.clone()
            rayf[:, :2] = ray[:, :2] / torch.exp(logf[0])
            Xw = C[refT] + torch.einsum('nij,nj->ni', R[refT], rayf * torch.exp(logz)[:, None])
            Pc = torch.einsum('nji,nj->ni', R[obs_t], Xw[obs_n] - C[obs_t])  # R^T (X - C)
            q = Pc @ Kf.T
            q = q[:, :2] / q[:, 2:].clamp_min(1e-3)
            e = torch.linalg.norm(q - obs_uv, dim=1)
            loss = torch.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber)).mean() \
                + prior * ((logz - logz0) ** 2).mean() * huber ** 2
            loss.backward()
            opt.step()
        with torch.no_grad():
            R = (_rotvec_to_R(dw * mask) @ R0).numpy()
            C = (C0 + dc * mask).numpy()
            f = float(torch.exp(logf[0]))
        new = c2w.copy()
        new[:, :3, :3], new[:, :3, 3] = R, C
        K = K.copy()
        K[0, 0] *= f
        K[1, 1] *= f
        Ki = np.linalg.inv(K)
        print(f'  BA round {rnd}: {len(cand)} tracks, {len(obs_t)} observations, median reprojection '
              f'{float(e.median()):.2f} px, focal x{f:.4f}', flush=True)
        c2w = new
    return c2w, K


def cameras(raw, depth):
    T = len(depth)
    sc = metric_scale(raw, depth)
    c2w = raw['mon_cam2world'].copy()
    c2w[:, :3, 3] *= sc
    c2w = interp_poses(raw['mon_sel'].astype(float), c2w, T)
    return c2w, raw['mon_K'], sc


# ---------------------------------------------------------------- track lifting (PO format)
def sample_depth(depth, t, uv, half=1, labels=None, lhalf=2):
    """Median z of the (2*half+1)^2 patch around each rounded pixel; nan off-image. depth (T, H, W).
    With instance labels (T, H, W), the median is taken over the (2*lhalf+1)^2 patch pixels that carry the same
    instance id as the point itself (>= 3 of them), so points next to an object boundary don't pick up the depth of
    the other side."""
    H, W = depth.shape[1:]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z = np.full(len(uv), np.nan, np.float32)
    patches = []
    for du in range(-half, half + 1):
        for dv in range(-half, half + 1):
            uu, vv = np.clip(u + du, 0, W - 1), np.clip(v + dv, 0, H - 1)
            patches.append(depth[t, vv, uu].astype(np.float32))
    z[ok] = np.median(np.stack(patches), 0)[ok]
    if labels is not None:
        uc, vc = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
        l0 = labels[t, vc, uc]
        vals, same = [], []
        for du in range(-lhalf, lhalf + 1):
            for dv in range(-lhalf, lhalf + 1):
                uu, vv = np.clip(u + du, 0, W - 1), np.clip(v + dv, 0, H - 1)
                vals.append(depth[t, vv, uu].astype(np.float32))
                same.append(labels[t, vv, uu] == l0)
        vals, same = np.stack(vals), np.stack(same)
        n = same.sum(0)
        zl = np.nanmedian(np.where(same, vals, np.nan), 0) if same.any() else z
        use = ok & (n >= 3)
        z[use] = zl[use]
    return z


def dyn_mask_frac(raw, uv, vis, T):
    """Per track: fraction of its visible MonST3R frames on which it lies in MonST3R's dynamic mask."""
    s, x0, y0, w, h = raw['mon_crop']
    hit, cnt = np.zeros(uv.shape[1]), np.zeros(uv.shape[1])
    for k, i in enumerate(raw['mon_sel']):
        x = np.round(uv[i, :, 0] * s - x0).astype(int)
        y = np.round(uv[i, :, 1] * s - y0).astype(int)
        ok = vis[i] & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        hit[ok] += raw['mon_dyn'][k][y[ok], x[ok]]
        cnt[ok] += 1
    return hit / np.maximum(cnt, 1)


# Static / dynamic decision per track: reprojection residual of the best static hypothesis. A static point must
# reproject onto its 2D track with the (accurate) cameras for some depth along its ray; checked on T3_001 against
# PO GT (world motion > 5 cm): recall 0.80 / precision 0.90 at 4 px, vs 0.38 / 0.92 for 3D spread + MonST3R mask.
RES_PX = 4.0  # at 960 px image width
FAR_STATIC = 25.0  # m: farther points are taken as static (no usable parallax, unreliable depth)
BORDER_PX = 10  # tracks whose median position is this close to the image border are taken as static
MIN_VIS_FRAMES, MIN_VIS_FRAC = 8, 0.1  # tracks visible on fewer frames are dropped
DEPTH_SEARCH = np.geomspace(0.6, 1.6, 21)  # depth scale factors tried along the ray


def static_residual(Xo, okv, uv, c2w, K):
    """Per track: 80th percentile over visible frames of the reprojection error of a static point placed at
    s * (median lifted depth) along the ray of its first visible frame, minimised over s."""
    T, N = okv.shape
    E = np.linalg.inv(c2w)
    first = np.argmax(okv, 0)
    C = c2w[first, :3, 3]
    med = np.nanmedian(np.where(okv[..., None], Xo, np.nan), 0)
    best = np.full(N, np.inf)
    for s in DEPTH_SEARCH:
        X = C + s * (med - C)
        P = np.einsum('tij,nj->tni', E[:, :3, :3], X) + E[:, None, :3, 3]
        q = P @ K.T
        q = q[..., :2] / np.maximum(q[..., 2:], 1e-3)
        e = np.where(okv & (P[..., 2] > 0.05), np.linalg.norm(q - uv, axis=-1), np.nan)
        best = np.minimum(best, np.nan_to_num(np.nanpercentile(e, 80, axis=0), nan=np.inf))
    return best


JUMP_M, JUMP_REL = 0.5, 0.15  # depth step between consecutive samples treated as a tracking failure


def smooth_visible(x, vis, cam, k=5, win=9, rel_floor=0.05):
    """Robust temporal filter over a track's visible samples; x (T, 3) world, vis (T,), cam (T, 3) camera centres.
    Hampel filter on the distance to the camera (window `win`, reflected at the clip ends; outlier if it deviates
    from the window median by > max(3 MAD, rel_floor * median): depth-edge bleeding), then a median (k) and a
    3-sample mean filter on the kept positions. Returns x and the kept mask."""
    from scipy.ndimage import median_filter, uniform_filter1d
    idx = np.nonzero(vis)[0]
    if len(idx) < 3:
        return x, vis
    d = np.linalg.norm(x[idx] - cam[idx], axis=1)
    w = min(win, len(idx) | 1)
    dm = median_filter(d, size=w, mode='reflect')
    mad = 1.4826 * median_filter(np.abs(d - dm), size=w, mode='reflect')
    ok = np.abs(d - dm) <= np.maximum(3 * mad, rel_floor * dm)
    idx, d = idx[ok], d[ok]
    if len(idx) >= 2:  # a surface point cannot jump in depth: split at jumps (tracker slid onto another surface)
        jump = (np.abs(np.diff(d)) > np.maximum(JUMP_M, JUMP_REL * d[:-1])) & (np.diff(idx) <= 3)
        seg = np.r_[0, np.cumsum(jump)]
        idx = idx[seg == np.bincount(seg).argmax()]  # keep the longest consistent segment
    vis = np.zeros_like(vis)
    vis[idx] = True
    if len(idx) >= 3:
        y = median_filter(x[idx], size=(min(k, len(idx)), 1), mode='reflect')
        x = x.copy()
        x[idx] = uniform_filter1d(y, 3, axis=0, mode='reflect')
    return x, vis


def lift_tracks(uv, vis, depth, c2w, K, raw, labels=None, qt=None):
    """2D tracks + metric depth + cameras -> world positions X (T, N, 3), dyn (N,), keep (N,).
    Static tracks get one world point (robust median over visible frames), so they move only with the camera;
    dynamic tracks are lifted per visible frame, smoothed, and linearly interpolated in world space through
    occlusions (held at the ends)."""
    T, N = vis.shape
    W = depth.shape[2]
    Ki = np.linalg.inv(K)
    Xo = np.full((T, N, 3), np.nan, np.float32)
    for t in range(T):
        z = sample_depth(depth, t, uv[t], labels=labels)
        ok = vis[t] & np.isfinite(z) & (z > 0.05)
        ray = np.c_[uv[t], np.ones(N)] @ Ki.T
        Pc = ray * z[:, None]
        Xw = Pc @ c2w[t, :3, :3].T + c2w[t, :3, 3]
        Xo[t, ok] = Xw[ok]
    okv = np.isfinite(Xo[..., 0])
    keep = okv.sum(0) >= max(MIN_VIS_FRAMES, MIN_VIS_FRAC * T)
    med = np.nanmedian(np.where(okv[..., None], Xo, np.nan), 0)
    res = static_residual(Xo, okv, uv, c2w, K)
    frac = dyn_mask_frac(raw, uv, okv, T)
    rng = np.nanmedian(np.where(okv, np.linalg.norm(Xo - c2w[:, None, :3, 3], axis=-1), np.nan), 0)
    H = depth.shape[1]
    mu = np.nanmedian(np.where(okv, uv[..., 0], np.nan), 0)
    mv = np.nanmedian(np.where(okv, uv[..., 1], np.nan), 0)
    inner = (mu >= BORDER_PX) & (mu < W - BORDER_PX) & (mv >= BORDER_PX) & (mv < H - BORDER_PX)
    dyn = (res >= RES_PX * W / 960) & (rng <= FAR_STATIC) & inner
    if labels is not None and qt is not None:  # one decision per object instance (object-sized regions only)
        lab = np.zeros(N, np.int32)
        for t in np.unique(qt):
            j = np.nonzero(qt == t)[0]
            u = np.clip(np.round(uv[t, j, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(uv[t, j, 1]).astype(int), 0, H - 1)
            area = np.bincount(labels[t].ravel(), minlength=256) / (H * W)
            l = labels[t][v, u]
            lab[j] = np.where((l > 0) & (area[l] <= 0.25), l, 0)
        for l in np.unique(lab[lab > 0]):
            m = (lab == l) & keep
            if m.sum() >= 5:
                dyn[lab == l] = dyn[m].mean() >= 0.5
    X = np.repeat(np.nan_to_num(med)[None], T, 0).astype(np.float32)
    t = np.arange(T)
    for j in np.nonzero(dyn & keep)[0]:
        x, ok = smooth_visible(Xo[:, j], okv[:, j], c2w[:, :3, 3])
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            keep[j] = False
            continue
        for c in range(3):
            X[:, j, c] = np.interp(t, idx, x[idx, c])
    return X, dyn, keep, frac, res


def warp_depth(z, c2w_src, E_dst, K, H, W):
    """Forward-splat depth map z of a source frame into a destination camera (nearest surface wins)."""
    v, u = np.mgrid[0:H, 0:W]
    ok = (z > 0.05) & (z < 200)
    Ki = np.linalg.inv(K)
    P = (np.c_[u[ok], v[ok], np.ones(ok.sum())] @ Ki.T) * z[ok, None]
    Xw = P @ c2w_src[:3, :3].T + c2w_src[:3, 3]
    Q = Xw @ E_dst[:3, :3].T + E_dst[:3, 3]
    q = Q @ K.T
    zz = q[:, 2]
    good = zz > 0.05
    uu, vv = np.round(q[good, 0] / zz[good]).astype(int), np.round(q[good, 1] / zz[good]).astype(int)
    zz = zz[good]
    m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    uu, vv, zz = uu[m], vv[m], zz[m]
    o = np.argsort(-zz)  # far first, so the nearest write wins
    out = np.full((H, W), np.nan, np.float32)
    out[vv[o], uu[o]] = zz[o]
    return out


def temporal_depth_filter(depth, c2w, K, offsets=(-8, -6, -4, -2, 2, 4, 6, 8), gate=0.08):
    """Multi-view median: each frame's depth is replaced by the median of itself and its neighbours' depth maps
    warped into it with the cameras, using only warped values within `gate` (relative) of the frame's own value, so
    moving objects and disocclusions keep their own depth."""
    T, H, W = depth.shape
    E = np.linalg.inv(c2w)
    out = np.empty_like(depth)
    for t in range(T):
        stack = [depth[t]]
        for o in offsets:
            n = t + o
            if 0 <= n < T:
                w = warp_depth(depth[n], c2w[n], E[t], K, H, W)
                w[~(np.abs(w - depth[t]) <= gate * depth[t])] = np.nan
                stack.append(w)
        out[t] = np.nanmedian(np.stack(stack), 0)
    return out


def depth_correction(depth, uv, vis, X, dyn, keep, E, K, iters=4):
    """Per frame, a smooth multiplicative correction exp(a + b x + c y) (x, y in [-1, 1]) that maps the depth map
    onto the fused static geometry: robust least squares on log(z_static_track / z_map) over static visible tracks.
    Removes VDA's per-frame scale / tilt flicker so the depth maps agree with the tracks as in GT."""
    T, H, W = depth.shape
    P = np.einsum('tij,tnj->tni', E[:, :3, :3], X) + E[:, None, :3, 3]
    yy, xx = np.mgrid[0:H, 0:W]
    xn, yn = xx / (W - 1) * 2 - 1, yy / (H - 1) * 2 - 1
    coef = np.zeros((T, 3))
    for t in range(T):
        u, v = np.round(uv[t, :, 0]).astype(int), np.round(uv[t, :, 1]).astype(int)
        ok = vis[t] & ~dyn & keep & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[t, :, 2] > 0.05)
        if ok.sum() < 20:
            continue
        zm = depth[t, v[ok], u[ok]]
        good = zm > 0.05
        lr = np.log(P[t, ok, 2][good] / zm[good])
        A = np.c_[np.ones(good.sum()), xn[v[ok], u[ok]][good], yn[v[ok], u[ok]][good]]
        w = np.ones(len(lr))
        for _ in range(iters):  # IRLS with Cauchy-like weights
            c = np.linalg.lstsq(A * w[:, None], lr * w, rcond=None)[0]
            r = lr - A @ c
            sig = 1.4826 * np.median(np.abs(r)) + 1e-4
            w = 1 / np.sqrt(1 + (r / (2 * sig)) ** 2)
        coef[t] = c
    from scipy.ndimage import uniform_filter1d
    coef = uniform_filter1d(coef, 3, axis=0, mode='nearest')  # corrections vary smoothly in time
    out = np.empty_like(depth)
    for t in range(T):
        out[t] = depth[t] * np.exp(coef[t, 0] + coef[t, 1] * xn + coef[t, 2] * yn)
    return out, coef


def fuse_po(raw, H, W):
    depth = raw['depth'].astype(np.float32)
    T = len(depth)
    c2w, K, sc = cameras(raw, depth)
    uv, vis = raw['ct_uv'], raw['ct_vis']
    c2w, K = bundle_adjust(uv, vis, depth, c2w, K)
    E = np.linalg.inv(c2w)
    n_grid = uv.shape[1]
    if 'ct_probe_uv' in raw:
        uv = np.concatenate([uv, raw['ct_probe_uv']], 1)
        vis = np.concatenate([vis, raw['ct_probe_vis']], 1)
        qt = np.r_[raw['ct_qt'], raw['probe_t']]
    else:
        qt = raw['ct_qt']
    X, dyn, keep, frac, resid = lift_tracks(uv, vis, depth, c2w, K, raw, raw['labels'], qt)
    for _ in range(2):  # make depth maps agree with the fused static geometry, then re-lift
        depth, coef = depth_correction(depth, uv, vis, X, dyn, keep, E, K)
        X, dyn, keep, frac, resid = lift_tracks(uv, vis, depth, c2w, K, raw, raw['labels'], qt)
    depth = temporal_depth_filter(depth, c2w, K)
    X, dyn, keep, frac, resid = lift_tracks(uv, vis, depth, c2w, K, raw, raw['labels'], qt)
    P = np.einsum('tij,tnj->tni', E[:, :3, :3], X) + E[:, None, :3, 3]
    # hidden static points: CoTracker can only guess where an occluded point is, while the bundle-adjusted cameras
    # place a static 3D point to < 1 px; use the reprojection there (GT uv is the projection of the 3D point too)
    q = P @ K.T
    proj = q[..., :2] / np.maximum(q[..., 2:], 1e-6)
    use = (~vis) & (~dyn)[None] & keep[None] & (P[..., 2] > 0.05)
    uv = np.where(use[..., None], proj, uv).astype(np.float32)
    inb = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
    valid = inb & (P[..., 2] > 0.05)
    return dict(c2w=c2w, E=E, K=K, scale=sc, uv=uv, vis=vis & valid, valid=valid, X=X, P=P, dyn=dyn, keep=keep,
                dyn_frac=frac, resid=resid, n_grid=n_grid, depth=depth)


# ---------------------------------------------------------------- writers
def write_blk(d, kind, items):
    """items: [(name, bytes)] -> <d>/<kind>.blk + <kind>.idx.json ([name, offset, size, size, 0], as pack_po.py)."""
    idx = []
    with open(f'{d}/{kind}.blk.tmp', 'wb') as f:
        for name, data in items:
            idx.append([name, f.tell(), len(data), len(data), 0])
            f.write(data)
    os.replace(f'{d}/{kind}.blk.tmp', f'{d}/{kind}.blk')
    json.dump(idx, open(f'{d}/{kind}.idx.json', 'w'))


def png_bytes(a, mode=None):
    b = io.BytesIO()
    Image.fromarray(a, mode=mode).save(b, format='PNG', compress_level=1)
    return b.getvalue()


def jpg_bytes(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format='JPEG', quality=95)
    return b.getvalue()


def write_po(name, frames, raw, meta):
    d = f'{PO_ROOT}/pseudo/{name}'
    os.makedirs(d, exist_ok=True)
    T, H, W = frames.shape[:3]
    f = fuse_po(raw, H, W)
    g = f['keep'][:f['n_grid']]  # grid tracks -> anno.npz; probe tracks (GT item points) -> probes.npz
    np.savez(f'{d}/anno.npz.tmp.npz', trajs_2d=f['uv'][:, :f['n_grid']][:, g].astype(np.float32),
             trajs_3d=f['X'][:, :f['n_grid']][:, g].astype(np.float32), visibs=f['vis'][:, :f['n_grid']][:, g],
             valids=f['valid'][:, :f['n_grid']][:, g], intrinsics=np.repeat(f['K'][None], T, 0).astype(np.float32),
             extrinsics=f['E'].astype(np.float32))
    os.replace(f'{d}/anno.npz.tmp.npz', f'{d}/anno.npz')
    p = slice(f['n_grid'], None)
    np.savez(f'{d}/probes.npz', uv=f['uv'][:, p], vis=f['vis'][:, p], valid=f['valid'][:, p], X=f['X'][:, p],
             P=f['P'][:, p], dyn=f['dyn'][p], keep=f['keep'][p], dyn_grid=f['dyn'][:f['n_grid']][g],
             resid_grid=f['resid'][:f['n_grid']][g], frac_grid=f['dyn_frac'][:f['n_grid']][g])
    write_blk(d, 'rgbs', [(f'rgb_{i:05d}.jpg', jpg_bytes(frames[i])) for i in range(T)])
    dep = np.clip(np.round(f['depth'] * (65535.0 / 1000.0)), 0, 65535).astype(np.uint16)
    write_blk(d, 'depths', [(f'depth_{i:05d}.png', png_bytes(dep[i])) for i in range(T)])
    write_blk(d, 'masks', [(f'mask_{i:05d}.png', png_bytes(raw['labels'][i], 'L')) for i in range(T)])
    json.dump(dict(meta, metric_scale=f['scale'], n_tracks=int(g.sum()), n_dyn=int(f['dyn'][:f['n_grid']][g].sum()),
                   K=f['K'].tolist()), open(f'{d}/pseudo_info.json', 'w'), indent=1)
    open(f'{d}/.complete', 'w').close()
    print('WROTE PO scene', d, f'{int(g.sum())} tracks', f"scale {f['scale']:.3f}", flush=True)


# KITTI layout: velodyne frame = KITTI convention (x fwd, y left, z up) at the camera centre (so the road is at
# z ~ -1.65 m, close to KITTI's -1.73 m that the T6 'ground' score term assumes); imu = velo
VELO_TO_CAM = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)


def oxts_from_poses(M, fps):
    """M (T, 4, 4) imu-to-world with M[0] = I (x fwd, y left, z up). Returns (T, 30) OXTS rows that the walkthrough's
    oxts_poses() maps back to M (lat0 = 0 so the Mercator scale is 1), with smoothed speed (vn, ve) and yaw rate wu."""
    from scipy.signal import savgol_filter
    er = 6378137.0
    T = len(M)
    O = np.zeros((T, 30))
    tx, ty, tz = M[:, 0, 3], M[:, 1, 3], M[:, 2, 3]
    O[:, 0] = 360 / np.pi * np.arctan(np.exp(ty / er)) - 90
    O[:, 1] = tx * 180 / (np.pi * er)
    O[:, 2] = tz
    R = M[:, :3, :3]
    O[:, 5] = np.arctan2(R[:, 1, 0], R[:, 0, 0])  # yaw
    O[:, 4] = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))  # pitch
    O[:, 3] = np.arctan2(R[:, 2, 1], R[:, 2, 2])  # roll
    win = min(T - (1 - T % 2), max(5, int(round(1.1 * fps)) | 1))  # ~1 s Savitzky-Golay window (odd, <= T)
    dt = 1.0 / fps
    vx = savgol_filter(tx, win, 2, deriv=1, delta=dt)
    vy = savgol_filter(ty, win, 2, deriv=1, delta=dt)
    O[:, 6], O[:, 7] = vy, vx  # vn, ve: only hypot(vn, ve) is used
    O[:, 22] = savgol_filter(np.unwrap(O[:, 5]), win, 2, deriv=1, delta=dt)  # wu, + = turning left
    O[:, 23:30] = [0.1, 0.1, 4, 10, 5, 5, 5]
    return O


def box_from_points(Xv):
    """Oriented box (KITTI tracklet: centre-bottom pose, h, w, l) around velodyne-frame points."""
    xy = Xv[:, :2] - np.median(Xv[:, :2], 0)
    ev, evec = np.linalg.eigh(np.cov(xy.T) + 1e-6 * np.eye(2))
    yaw = float(np.arctan2(evec[1, 1], evec[0, 1]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    q = Xv[:, :2]
    a, b = c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]
    lo_a, hi_a = np.percentile(a, [2, 98])
    lo_b, hi_b = np.percentile(b, [2, 98])
    lo_z, hi_z = np.percentile(Xv[:, 2], [2, 98])
    ca, cb = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    cx, cy = np.cos(yaw) * ca - np.sin(yaw) * cb, np.sin(yaw) * ca + np.cos(yaw) * cb
    return [cx, cy, lo_z, 0.0, 0.0, yaw], hi_z - lo_z, hi_b - lo_b, hi_a - lo_a


def tracklet_xml(raw, depth, K):
    """One single-frame tracklet per detected object mask: the mask (eroded) is lifted with the depth map and
    boxed in velodyne coordinates."""
    import cv2
    T, H, W = depth.shape
    masks = np.unpackbits(raw['det_mask'], axis=-1)[..., :W].astype(bool) if raw['det_frame'].size else []
    Ki = np.linalg.inv(K)
    items = []
    for f, ty, m in zip(raw['det_frame'], raw['det_type'], masks):
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        v, u = np.nonzero(m)
        if len(u) < 30:
            continue
        z = depth[f, v, u].astype(np.float64)
        zm = np.median(z)
        ok = (z > 0.5 * zm) & (z < 1.5 * zm) & (z < 80)
        if ok.sum() < 30:
            continue
        Xc = (np.c_[u[ok], v[ok], np.ones(ok.sum())] @ Ki.T) * z[ok, None]
        Xv = Xc @ VELO_TO_CAM  # cam -> velo (VELO_TO_CAM is orthonormal)
        pose, h, w, l = box_from_points(Xv)
        items.append(f'<item><objectType>{ty}</objectType><h>{h:.4f}</h><w>{w:.4f}</w><l>{l:.4f}</l>'
                     f'<first_frame>{int(f)}</first_frame><poses><count>1</count><item>'
                     + ''.join(f'<{k}>{x:.6f}</{k}>' for k, x in zip(('tx', 'ty', 'tz', 'rx', 'ry', 'rz'), pose))
                     + '</item></poses></item>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<boost_serialization><tracklets>'
            f'<count>{len(items)}</count>' + ''.join(items) + '</tracklets></boost_serialization>\n')


VELO_ELEV = (-24.9, 2.0)  # deg, vertical field of view of KITTI's HDL-64E


def pseudo_lidar(z, K, stride=2, z_max=80.0):
    """Velodyne-format (N, 4) float32 scan from a metric depth map: every stride-th pixel, z in (0.5, z_max), only
    inside the HDL-64E vertical field of view (so points are placed where the real sensor has returns)."""
    H, W = z.shape
    v, u = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    zz = z[v, u].astype(np.float64)
    ok = (zz > 0.5) & (zz < z_max)
    Xc = (np.c_[u[ok], v[ok], np.ones(ok.sum())] @ np.linalg.inv(K).T) * zz[ok, None]
    Xv = Xc @ VELO_TO_CAM
    el = np.degrees(np.arctan2(Xv[:, 2], np.hypot(Xv[:, 0], Xv[:, 1])))
    Xv = Xv[(el >= VELO_ELEV[0]) & (el <= VELO_ELEV[1])]
    return np.c_[Xv, np.zeros(len(Xv))].astype(np.float32)


def vo_cameras(raw, depth, K, step=(1, 2, 3)):
    """Metric ego-motion for driving clips: relative pose frame t -> t+k by PnP-RANSAC on CoTracker correspondences,
    3D from the metric depth at t (static points only: detected traffic participants are excluded). The relative
    motions for k = 1..3 are combined (median of the per-frame translation implied by each k) and chained.
    MonST3R's translation is poorly constrained on KITTI's wide frames (speed off by 2x), its rotation is not."""
    import cv2
    from scipy.spatial.transform import Rotation
    T, H, W = depth.shape
    uv, vis = raw['ct_uv'], raw['ct_vis']
    obj = np.zeros((T, H, W), bool)
    if raw['det_frame'].size:
        masks = np.unpackbits(raw['det_mask'], axis=-1)[..., :W].astype(bool)
        for f, m in zip(raw['det_frame'], masks):
            obj[f] |= cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    Ki = np.linalg.inv(K)
    rel = {}
    for k in step:
        for t in range(T - k):
            ok = vis[t] & vis[t + k]
            u, v = np.round(uv[t, :, 0]).astype(int), np.round(uv[t, :, 1]).astype(int)
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ok[ok] &= ~obj[t, v[ok], u[ok]]
            z = np.where(ok, depth[t, np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], 0)
            ok &= (z > 1) & (z < 40)
            if ok.sum() < 12:
                continue
            P = (np.c_[uv[t, ok], np.ones(ok.sum())] @ Ki.T) * z[ok, None]
            good, rv, tv, inl = cv2.solvePnPRansac(P.astype(np.float64), uv[t + k, ok].astype(np.float64), K, None,
                                                   reprojectionError=2.0, iterationsCount=200,
                                                   flags=cv2.SOLVEPNP_EPNP)
            if good and inl is not None and len(inl) >= 12:
                rv, tv = cv2.solvePnPRefineLM(P[inl[:, 0]], uv[t + k, ok][inl[:, 0]].astype(np.float64), K, None, rv, tv)
                M = np.eye(4)
                M[:3, :3] = cv2.Rodrigues(rv)[0]
                M[:3, 3] = tv.ravel()
                rel[(t, k)] = M  # cam_t -> cam_{t+k}
    # per-step motion t -> t+1: rotation from k = 1, translation = median over k of (t_k / k) expressed per step
    E = [np.eye(4)]
    for t in range(T - 1):
        if (t, 1) in rel:
            M = rel[(t, 1)].copy()
            ts = [rel[(t, 1)][:3, 3]] + [rel[(t, k)][:3, 3] / k for k in step[1:] if (t, k) in rel]
            M[:3, 3] = np.median(np.array(ts), 0)
        else:  # no estimate: repeat the previous step
            M = E[-1] @ np.linalg.inv(E[-2]) if len(E) > 1 else np.eye(4)
        E.append(M @ E[-1])  # world (= cam 0) -> cam t+1
    E = np.array(E)
    print(f'  VO: {sum(1 for (t, k) in rel if k == 1)}/{T - 1} consecutive pairs solved', flush=True)
    return np.linalg.inv(E)


def write_kitti(drive, frames, fps, raw, meta):
    date = drive[:10]
    dd = f'{KITTI_ROOT}/{date}/{drive}_sync'
    os.makedirs(f'{dd}/oxts/data', exist_ok=True)
    T, H, W = frames.shape[:3]
    depth = raw['depth'].astype(np.float32)
    _, K, sc = cameras(raw, depth)
    c2w = vo_cameras(raw, depth, K)
    fmt = lambda a: ' '.join(f'{x:.12e}' for x in np.ravel(a))
    P2 = np.c_[K, np.zeros(3)]
    with open(f'{KITTI_ROOT}/{date}/calib_cam_to_cam.txt', 'w') as f:
        f.write(f'S_rect_02: {W} {H}\nR_rect_00: {fmt(np.eye(3))}\nP_rect_02: {fmt(P2)}\n')
    with open(f'{KITTI_ROOT}/{date}/calib_velo_to_cam.txt', 'w') as f:
        f.write(f'R: {fmt(VELO_TO_CAM)}\nT: {fmt(np.zeros(3))}\n')
    with open(f'{KITTI_ROOT}/{date}/calib_imu_to_velo.txt', 'w') as f:
        f.write(f'R: {fmt(np.eye(3))}\nT: {fmt(np.zeros(3))}\n')
    T_cam_velo = np.eye(4)
    T_cam_velo[:3, :3] = VELO_TO_CAM
    Mw = c2w @ T_cam_velo  # velo(=imu)-to-world, MonST3R world
    M = np.linalg.inv(Mw[0]) @ Mw
    O = oxts_from_poses(M, fps)
    for i in range(T):
        np.savetxt(f'{dd}/oxts/data/{i:010d}.txt', O[i][None], fmt='%.12g')
    base = 13 * 3600
    with open(f'{dd}/image_02_timestamps.txt', 'w') as f:
        for i in range(T):
            s = base + i / fps
            f.write(f'2000-01-01 {int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:012.9f}\n')
    write_blk(dd, 'image_02', [(f'{i:010d}.png', png_bytes(frames[i])) for i in range(T)])
    write_blk(dd, 'velodyne', [(f'{i:010d}.bin', pseudo_lidar(depth[i], K).tobytes()) for i in range(T)])
    open(f'{dd}/tracklet_labels.xml', 'w').write(tracklet_xml(raw, depth, K))
    json.dump(dict(meta, metric_scale=sc, K=K.tolist()), open(f'{dd}/pseudo_info.json', 'w'), indent=1)
    open(f'{dd}/.blk_complete', 'w').close()
    print('WROTE KITTI drive', dd, f'scale {sc:.3f}', 'max speed %.1f m/s' % np.hypot(O[:, 6], O[:, 7]).max(), flush=True)


# ---------------------------------------------------------------- entry points
def load_raw(name):
    z = np.load(f'{RAW}/{name}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def save_raw(name, raw):
    os.makedirs(RAW, exist_ok=True)
    np.savez(f'{RAW}/{name}.tmp.npz', **raw)
    os.replace(f'{RAW}/{name}.tmp.npz', f'{RAW}/{name}.npz')


def clip_frames(c):
    if c['src'] == 'po':
        return frames_po(c['scene'], c['start'], c['end'])
    if c['src'] == 'kitti':
        return frames_kitti(c['drive'], c['start'], c['end'])
    return frames_video(c['path'], None if c.get('kitti') else PO_FPS)


QWEN_PY = '/shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python'


def detect_boxes_subprocess(c):
    """Grounding DINO needs a newer torch than the specialist env: run `build.py detect` in the qwen_video env."""
    import subprocess
    path = f"{RAW}/{c['name']}_boxes.json"
    if not os.path.exists(path):
        subprocess.run([QWEN_PY, os.path.abspath(__file__), 'detect', json.dumps(c)], check=True)
    return load_boxes(path)


def load_boxes(path):
    return [(b['types'], np.array(b['boxes'], np.float32).reshape(-1, 4)) for b in json.load(open(path))]


def start_detect(c):
    """`build.py detect` as a child (on PGT_CHILD_GPU if set), alongside VDA / MonST3R. None if the boxes are cached."""
    import subprocess
    path = f"{RAW}/{c['name']}_boxes.json"
    if os.path.exists(path):
        return None
    return subprocess.Popen([QWEN_PY, os.path.abspath(__file__), 'detect', json.dumps(c)], env=child_env()), path


def finish_detect(proc, path):
    if proc.wait() != 0:
        raise RuntimeError(f'detector child failed (exit {proc.returncode})')
    return load_boxes(path)


def redo(raw, stages, frames, fps, c):
    """Recompute some specialist stages of a cached raw output (e.g. after changing the CoTracker settings)."""
    import specialists as S
    if 'cotracker' in stages:
        probes = np.array([[p['t'], p['u'], p['v']] for p in c.get('probes', [])], np.float32)
        ct = S.run_cotracker(frames, fps, extra=probes)
        raw.update({f'ct_{k}': v for k, v in ct.items()})
    if 'sam2' in stages:
        raw['labels'] = S.run_sam2_instances(frames, fps)
    if 'vda' in stages:
        raw['depth'] = S.run_vda(frames, fps).astype(np.float16)
    save_raw(c['name'], raw)
    return raw


def build_clip(c, fuse_only=False, raw_only=False, redo_stages=(), parallel=False):
    frames, fps = clip_frames(c)
    kitti = c['src'] == 'kitti' or c.get('kitti', False)
    if redo_stages and os.path.exists(f"{RAW}/{c['name']}.npz"):
        raw = redo(load_raw(c['name']), redo_stages, frames, fps, c)
    elif fuse_only and os.path.exists(f"{RAW}/{c['name']}.npz"):
        raw = load_raw(c['name'])
    else:
        probes = np.array([[p['t'], p['u'], p['v']] for p in c.get('probes', [])], np.float32) if not kitti else None
        det_job = start_detect(c) if kitti and parallel else None
        boxes = (load_boxes(f"{RAW}/{c['name']}_boxes.json") if det_job is None else None) if kitti and parallel \
            else (detect_boxes_subprocess(c) if kitti else None)
        raw = run_specialists(frames, fps, kitti=kitti, probes=probes, boxes=boxes, parallel=parallel, det_job=det_job)
        if probes is not None and len(probes):
            raw['probe_t'] = probes[:, 0].astype(np.int32)
        save_raw(c['name'], raw)
    if raw_only:
        return
    meta = {k: v for k, v in c.items() if k != 'probes'}
    meta['fps'] = fps
    if kitti:
        write_kitti(c['drive_id'], frames, fps, raw, meta)
    else:
        write_po(c['name'], frames, raw, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['clip', 'video', 'detect', 'sam2'])
    ap.add_argument('arg', help='clip name from clips.json, or a video path')
    ap.add_argument('--name', default='')
    ap.add_argument('--kitti', action='store_true', help='video mode: write the KITTI layout (T6) instead of PO')
    ap.add_argument('--fuse_only', action='store_true', help='reuse the cached specialist outputs')
    ap.add_argument('--raw_only', action='store_true', help='only run the specialists (condor jobs on other hosts)')
    ap.add_argument('--redo', default='', help='comma list of cached stages to recompute: cotracker,sam2,vda')
    ap.add_argument('--parallel', action='store_true', help='run SAM2 (KITTI: the detector) in a child process alongside '
                    'the other GPU stages, on $PGT_CHILD_GPU if set')
    ap.add_argument('--fps', type=float, default=0, help='internal (sam2 mode)')
    a = ap.parse_args()
    if a.mode == 'detect':  # internal: qwen_video env, writes <RAW>/<name>_boxes.json
        import specialists as S
        c = json.loads(a.arg)
        frames, _ = clip_frames(c)
        det = S.detect_boxes(frames)
        json.dump([dict(types=t, boxes=b.tolist()) for t, b in det], open(f"{RAW}/{c['name']}_boxes.json", 'w'))
        print('DETECT DONE', c['name'], sum(len(t) for t, _ in det), 'boxes', flush=True)
        sys.exit(0)
    if a.mode == 'sam2':  # internal: SAM2 child of run_specialists(parallel=True); arg = job dir with frames.npy
        import specialists as S
        t = time.time()
        labels = S.run_sam2_instances(np.load(f'{a.arg}/frames.npy'), a.fps)
        np.save(f'{a.arg}/labels.npy', labels)
        print(f'SAM2 child {time.time() - t:.0f} s ({labels.max()} instances, peak {gpu_peak():.1f} GiB)', flush=True)
        sys.exit(0)
    if a.mode == 'clip':
        c = next(c for c in json.load(open(CLIPS)) if c['name'] == a.arg)
    else:
        name = a.name or os.path.splitext(os.path.basename(a.arg))[0]
        c = dict(name=name, src='video', path=os.path.abspath(a.arg), kitti=a.kitti,
                 drive_id=f'v{zlib.crc32(name.encode()) % 10 ** 8:08d}_{name}' if a.kitti else None)
    t = time.time()
    build_clip(c, a.fuse_only, a.raw_only, [x for x in a.redo.split(',') if x], a.parallel)
    print(f'BUILD DONE {c["name"]} in {time.time() - t:.0f} s', flush=True)
