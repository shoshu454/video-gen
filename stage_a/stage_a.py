# Stage A: fill in the missing geometry of standard-format videos with the specialist models.
#   python stage_a.py <input_dir> <output_dir> [--segment 300] [--shard 0/4] ...
# Input and output are both in the standard format (README_FORMAT.md). For each video, GT modalities are hard-linked
# (or copied) to the output unchanged; missing ones are estimated in this order, reusing ../specialist:
#   1. intrinsics  MonST3R focal (refined by the bundle adjustment of step 4 when the extrinsics are estimated too)
#   2. masks       SAM2.1 automatic instances + video propagation (specialists.run_sam2_instances), named with
#                  Grounding DINO boxes (specialists.detect_boxes, in the qwen_video env: gdino_server.py)
#   3. depth       Video Depth Anything, metric ViT-L (specialists.run_vda); with GT extrinsics, corrected per frame to
#                  static tracks triangulated with them (build.depth_correction)
#   4. extrinsics  MonST3R global alignment + bundle adjustment on the 2D tracks, scaled to the metric depth
#                  (build.cameras / build.bundle_adjust), or --pose_method vo: PnP visual odometry (build.vo_cameras)
#   5. tracks      CoTracker3 (specialists.track_queries), queries every 12 frames on a grid inside masks and on the
#                  background, away from depth edges; lifted to 3D with depth + cameras (build.lift_tracks)
# The 2D tracks are computed before step 4 when the extrinsics are estimated, because the bundle adjustment uses them.
# SAM2 and CoTracker run in segments of --segment frames (overlap --overlap) and are stitched; VDA is windowed
# internally and MonST3R uses <= --monst3r_frames frames, so both run on the whole video.
# Resume: <out>/<video>/DONE is written last; finished modalities are skipped; intermediate model outputs are cached
# in <out>/<video>/_work/ (deleted when the video is done). Failures go to <out>/failures.jsonl.
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE: THP compaction stalls big allocations on vision-c22
import argparse, datetime, importlib.util, json, os, shutil, socket, subprocess, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(os.path.dirname(HERE), 'specialist')
sys.path.insert(0, HERE)
import stdfmt as F

QWEN_PY = os.environ.get('PVIDEO_GDINO_PYTHON', '/shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python')
VOCAB = ('person, car, truck, bus, bicycle, motorcycle, dog, cat, horse, bird, chair, table, sofa, bed, bottle, cup, '
         'bowl, laptop, keyboard, phone, book, ball, box, bag, toy, plant, door, window, cabinet, shelf, tree, '
         'building, floor, wall, ceiling, ground, road, grass, sky')
VERSION = 'stage_a 1.0'
S = B = None  # specialists / build modules, imported lazily (torch, CUDA)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_specialists(scratch):
    global S, B
    if S is None:
        os.environ['PGT_SCRATCH'] = scratch  # specialists.py reads it at import time
        sys.path.insert(0, SPEC)
        import specialists
        spec = importlib.util.spec_from_file_location('pgt_build', os.path.join(SPEC, 'build.py'))
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        S, B = specialists, build


def free_gpu():
    import torch
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- Grounding DINO child (qwen_video env)
class GDino:
    def __init__(self, python, scratch):
        self.python, self.scratch, self.proc = python, scratch, None

    def _start(self):
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.proc = subprocess.Popen([self.python, os.path.join(HERE, 'gdino_server.py')], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)
        line = self.proc.stdout.readline()
        if line.strip() != 'READY':
            raise RuntimeError(f'Grounding DINO worker did not start (got {line!r}, exit {self.proc.poll()})')

    def detect(self, frames, vocab):
        """frames (n, H, W, 3) -> [(names, (m, 4) xyxy boxes)] per frame."""
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        os.makedirs(self.scratch, exist_ok=True)
        base = os.path.join(self.scratch, f'gdino_{os.getpid()}')
        np.save(base + '.npy', frames)
        self.proc.stdin.write(json.dumps(dict(frames=base + '.npy', vocab=vocab, out=base + '.json')) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line.startswith('DONE'):
            raise RuntimeError(f'Grounding DINO failed: {line.strip() or "worker died"}')
        res = json.load(open(base + '.json'))
        for e in ('.npy', '.json'):
            os.remove(base + e)
        return [(r['types'], np.array(r['boxes'], np.float32).reshape(-1, 4)) for r in res]

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------- segments
def segments(T, length, overlap):
    """[(start, end)) windows of `length` frames overlapping by `overlap`, covering 0..T."""
    if T <= length:
        return [(0, T)]
    out, s = [], 0
    while True:
        e = min(s + length, T)
        out.append((s, e))
        if e == T:
            return out
        s = e - overlap


# ---------------------------------------------------------------- masks
def sam2_segments(frames, fps, segs):
    """specialists.run_sam2_instances per segment; ids of consecutive segments are matched on the overlap frames
    (IoU >= 0.5, one-to-one, largest IoU first). Returns (T, H, W) uint16, 0 = background."""
    T, H, W = frames.shape[:3]
    out = np.zeros((T, H, W), np.uint16)
    next_id, prev_end = 1, 0
    for k, (s, e) in enumerate(segs):
        log(f'  SAM2 segment {k + 1}/{len(segs)}: frames {s}..{e - 1}')
        lab = S.run_sam2_instances(frames[s:e], fps).astype(np.int64)
        free_gpu()
        ids = [i for i in np.unique(lab) if i > 0]
        lut = np.zeros(lab.max() + 1, np.int64)
        ov = range(s, prev_end)
        pairs = []
        if len(ov):
            g = out[s:prev_end].astype(np.int64)
            l = lab[:prev_end - s]
            for i in ids:
                mi = l == i
                if not mi.any():
                    continue
                cand, cnt = np.unique(g[mi], return_counts=True)
                for gid, inter in zip(cand, cnt):
                    if gid > 0:
                        union = mi.sum() + (g == gid).sum() - inter
                        pairs.append((inter / union, i, gid))
        used_l, used_g = set(), set()
        for iou, i, gid in sorted(pairs, reverse=True):
            if iou >= 0.5 and i not in used_l and gid not in used_g:
                lut[i] = gid
                used_l.add(i)
                used_g.add(gid)
        for i in ids:
            if i not in used_l:
                lut[i] = next_id
                next_id += 1
        mapped = lut[lab]
        cut = s + (prev_end - s) // 2  # overlap: first half from the previous segment, second half from this one
        out[cut:e] = mapped[cut - s:]
        prev_end = e
    assert next_id < 65536
    return out


def name_instances(masks, frames, fps, gdino, vocab):
    """{instance id: name}: Grounding DINO boxes on key frames (every 2 s) vote for an instance when the box matches its
    mask: box IoU with the mask's bounding box >= 0.5, or >= 70% of the mask inside the box with the mask filling >= 30%
    of it (irregular shapes). Votes are weighted by the match score; instances without votes are 'unknown'."""
    T = len(masks)
    keys = list(range(0, T, max(1, int(round(2 * fps)))))
    det = gdino.detect(frames[keys], vocab)
    votes = {}
    for t, (types, boxes) in zip(keys, det):
        lab = masks[t]
        for i in np.unique(lab):
            if i == 0 or not len(boxes):
                continue
            v, u = np.nonzero(lab == i)
            mb = np.array([u.min(), v.min(), u.max() + 1, v.max() + 1], np.float32)
            ix0, iy0 = np.maximum(mb[0], boxes[:, 0]), np.maximum(mb[1], boxes[:, 1])
            ix1, iy1 = np.minimum(mb[2], boxes[:, 2]), np.minimum(mb[3], boxes[:, 3])
            inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
            iou = inter / (area(mb) + area(boxes) - inter + 1e-6)
            inside = np.array([((u >= b[0]) & (u < b[2]) & (v >= b[1]) & (v < b[3])).mean() for b in boxes])
            fill = len(u) / np.maximum(area(boxes), 1)
            score = np.where(iou >= 0.5, iou, np.where((inside >= 0.7) & (fill >= 0.3), 0.5 * inside, 0))
            j = int(np.argmax(score))
            if score[j] > 0:
                votes.setdefault(int(i), {}).setdefault(types[j], 0.0)
                votes[int(i)][types[j]] += float(score[j])
    return {int(i): (max(votes[int(i)], key=votes[int(i)].get) if int(i) in votes else 'unknown')
            for i in np.unique(masks) if i > 0}


# ---------------------------------------------------------------- depth scale vs GT cameras
def triangulate_static(uv, vis, c2w, K, min_obs=5, px=2.0, min_angle_deg=1.0, chunk=4000):
    """Static-point triangulation of 2D tracks with known cameras (no depth needed): per track, the least-squares
    intersection of its visible rays. A track is static when it has >= min_obs observations, rays spanning
    >= min_angle_deg, positive depth everywhere and an 80th-percentile reprojection error <= px (per 960 px width).
    Returns X (N, 3) world points and the static mask (N,)."""
    T, N = vis.shape
    W = 2 * K[0, 2]
    Ki = np.linalg.inv(K)
    E = np.linalg.inv(c2w)
    C = c2w[:, :3, 3]
    X = np.full((N, 3), np.nan)
    ok = np.zeros(N, bool)
    for j0 in range(0, N, chunk):
        sl = slice(j0, min(N, j0 + chunk))
        v = vis[:, sl]
        r = np.concatenate([uv[:, sl], np.ones(v.shape + (1,))], -1) @ Ki.T
        d = np.einsum('tij,tnj->tni', c2w[:, :3, :3], r)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        M = np.eye(3) - d[..., :, None] * d[..., None, :]  # (T, n, 3, 3) projector orthogonal to each ray
        M *= v[..., None, None]
        A = M.sum(0)
        bb = np.einsum('tnij,tj->ni', M, C)
        nobs = v.sum(0)
        good = nobs >= min_obs
        good &= np.linalg.cond(A) < 1e6
        Xc = np.full((v.shape[1], 3), np.nan)
        Xc[good] = np.linalg.solve(A[good], bb[good][..., None])[..., 0]
        mean_d = np.nansum(np.where(v[..., None], d, 0), 0)
        mean_d /= np.linalg.norm(mean_d, axis=-1, keepdims=True) + 1e-12
        ang = np.degrees(np.arccos(np.clip(np.where(v, np.einsum('tni,ni->tn', d, mean_d), 1), -1, 1))).max(0)
        P = np.einsum('tij,nj->tni', E[:, :3, :3], np.nan_to_num(Xc)) + E[:, None, :3, 3]
        q = P @ K.T
        e = np.hypot(q[..., 0] / np.maximum(q[..., 2], 1e-9) - uv[:, sl, 0], q[..., 1] / np.maximum(q[..., 2], 1e-9) - uv[:, sl, 1])
        e = np.where(v, e, np.nan)
        res = np.nanpercentile(np.where(good[None], e, 0), 80, axis=0)
        front = np.where(v, P[..., 2] > 0.05, True).all(0)
        ok[sl] = good & (ang >= min_angle_deg) & front & (res <= px * W / 960)
        X[sl] = Xc
    return X, ok


def align_depth_to_cameras(depth, uv, vis, c2w, K, min_support=20):
    """Scale estimated depth to the GT (metric) cameras: static tracks are triangulated with the cameras, then
    build.depth_correction fits a smooth per-frame correction exp(a + b x + c y) of each depth map to them (the same
    fusion step the specialist pipeline uses). Frames with < min_support static tracks get the correction
    interpolated from their neighbours. Returns (corrected depth, info)."""
    T, H, W = depth.shape
    X, st = triangulate_static(uv, vis, c2w, K)
    if st.sum() < 50:
        return depth, dict(status=f'not aligned: only {int(st.sum())} static triangulated tracks', factor=1.0)
    Xs = np.repeat(X[None].astype(np.float32), T, 0)
    E = np.linalg.inv(c2w)
    _, coef = B.depth_correction(depth, uv, vis, Xs, ~st, st, E, K)
    P = np.einsum('tij,nj->tni', E[:, :3, :3], X[st]) + E[:, None, :3, 3]
    q = P @ K.T
    u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    support = (vis[:, st] & (u >= 0) & (u < W) & (v >= 0) & (v < H)).sum(1)
    fit = support >= min_support
    if not fit.any():
        return depth, dict(status='not aligned: no frame with enough static tracks', factor=1.0)
    t = np.arange(T)
    coef = np.stack([np.interp(t, t[fit], coef[fit, k]) for k in range(3)], 1)
    yy, xx = np.mgrid[0:H, 0:W]
    xn, yn = xx / (W - 1) * 2 - 1, yy / (H - 1) * 2 - 1
    out = np.stack([depth[i] * np.exp(coef[i, 0] + coef[i, 1] * xn + coef[i, 2] * yn) for i in range(T)])
    sc = np.exp(coef[:, 0])
    return out.astype(np.float32), dict(status='ok', factor=round(float(np.median(sc)), 5),
                                        factor_range=[round(float(sc.min()), 5), round(float(sc.max()), 5)],
                                        static_tracks=int(st.sum()), frames_fit=int(fit.sum()),
                                        method='triangulated static tracks + per-frame log-plane correction')


# ---------------------------------------------------------------- tracks
def depth_edges(z, ratio=1.1, size=5):
    """Pixels whose (size x size) neighbourhood spans a depth ratio > `ratio`, plus invalid depth."""
    from scipy.ndimage import maximum_filter, minimum_filter
    zz = np.where(z > 0, z, np.nan)
    mx = maximum_filter(np.nan_to_num(zz, nan=0.0), size)
    mn = minimum_filter(np.nan_to_num(zz, nan=np.inf), size)
    return ~(z > 0) | ~(mx <= ratio * mn)


def sample_queries(t, depth, masks, uv, vis, spacing, min_per_obj, rng):
    """Query pixels (n, 2) at frame t: a `spacing` grid away from depth edges, minus grid points within 0.7 * spacing
    of an existing visible track, plus random interior points for instances that would have < min_per_obj tracks."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    H, W = depth.shape[1:]
    xs, ys = np.arange(spacing / 2, W, spacing), np.arange(spacing / 2, H, spacing)
    gx, gy = [g.ravel() for g in np.meshgrid(xs, ys)]
    edge = depth_edges(depth[t])
    ok = ~edge[gy.astype(int), gx.astype(int)]
    have = uv[t][vis[t]] if uv.shape[1] else np.zeros((0, 2))
    if len(have):
        ok &= cKDTree(have).query(np.stack([gx, gy], 1))[0] > 0.7 * spacing
    q = np.stack([gx[ok], gy[ok]], 1)
    lab = masks[t]
    hu = np.clip(np.round(have).astype(int), 0, [W - 1, H - 1]) if len(have) else np.zeros((0, 2), int)
    have_lab = lab[hu[:, 1], hu[:, 0]] if len(hu) else np.zeros(0, lab.dtype)
    q_lab = lab[q[:, 1].astype(int), q[:, 0].astype(int)] if len(q) else np.zeros(0, lab.dtype)
    extra = []
    for i in np.unique(lab):
        if i == 0:
            continue
        n = (have_lab == i).sum() + (q_lab == i).sum()
        if n >= min_per_obj:
            continue
        m = binary_erosion(lab == i, iterations=2) & ~edge
        v, u = np.nonzero(m)
        if len(u) < 20:
            continue
        j = rng.choice(len(u), min(min_per_obj - n, len(u)), replace=False)
        extra.append(np.stack([u[j], v[j]], 1).astype(np.float64))
    if extra:
        q = np.concatenate([q] + extra)
    return q.astype(np.float32)


def run_tracks(frames, depth, masks, segs, every=12, spacing=None, min_per_obj=8, max_side=960, seed=0):
    """CoTracker3 tracks over the whole video. Per segment: tracks still visible in the overlap with the previous
    segment are continued (re-queried at their last visible overlap frame); then new queries every `every` frames
    (each query frame belongs to the first segment that contains it). CoTracker sees frames downscaled to
    max_side; coordinates are in full-resolution pixels. Returns uv (T, N, 2), vis (T, N), qt (N,)."""
    import cv2
    T, H, W = frames.shape[:3]
    spacing = spacing or max(8, round(12 * max(H, W) / 960))
    f = min(1.0, max_side / max(H, W))
    to_s = lambda p: (p + 0.5) * f - 0.5
    from_s = lambda p: (p + 0.5) / f - 0.5
    rng = np.random.default_rng(seed)
    uv = np.zeros((T, 0, 2), np.float32)
    vis = np.zeros((T, 0), bool)
    qt = np.zeros(0, np.int64)
    known = np.zeros((T, 0), bool)  # frames where a track has a position (its segments)
    done_q = set()
    prev_end = 0
    for k, (s, e) in enumerate(segs):
        fs = frames[s:e] if f == 1 else np.stack([cv2.resize(x, (int(round(W * f)), int(round(H * f))),
                                                             interpolation=cv2.INTER_AREA) for x in frames[s:e]])
        if prev_end > s:  # continue tracks through the overlap
            cont = np.nonzero(vis[s:prev_end].any(0))[0]
            if len(cont):
                last = s + (prev_end - s - 1) - np.argmax(vis[s:prev_end][::-1, cont], 0)
                q = np.c_[last - s, to_s(uv[last, cont])].astype(np.float32)
                u, v = S.track_queries(fs, q)
                for n, j in enumerate(cont):
                    a = last[n] - s
                    uv[s + a:e, j] = from_s(u[a:, n])
                    vis[s + a:e, j] = v[a:, n]
                    known[s + a:e, j] = True
                log(f'  CoTracker segment {k + 1}/{len(segs)}: continued {len(cont)} tracks from frame {s}')
        for t in range(s, e, 1):
            if t % every or t in done_q:
                continue
            done_q.add(t)
            q = sample_queries(t, depth, masks, uv, vis, spacing, min_per_obj, rng)
            if len(q) < 10:
                continue
            u, v = S.track_queries(fs, np.c_[np.full(len(q), t - s), to_s(q)].astype(np.float32))
            n = len(q)
            uv = np.concatenate([uv, np.zeros((T, n, 2), np.float32)], 1)
            vis = np.concatenate([vis, np.zeros((T, n), bool)], 1)
            known = np.concatenate([known, np.zeros((T, n), bool)], 1)
            uv[s:e, -n:] = from_s(u)
            vis[s:e, -n:] = v
            known[s:e, -n:] = True
            qt = np.r_[qt, np.full(n, t)]
            log(f'  CoTracker: query frame {t}: +{n} tracks ({uv.shape[1]} total)')
        prev_end = e
        free_gpu()
    # frames outside a track's segments: hold the nearest known position (invisible)
    t_idx = np.arange(T)
    for j in np.nonzero(~known.all(0))[0]:
        idx = np.nonzero(known[:, j])[0]
        near = idx[np.clip(np.searchsorted(idx, t_idx), 0, len(idx) - 1)]
        prev = idx[np.clip(np.searchsorted(idx, t_idx) - 1, 0, len(idx) - 1)]
        near = np.where(np.abs(prev - t_idx) <= np.abs(near - t_idx), prev, near)
        uv[:, j] = uv[near, j]
    return uv.astype(np.float32), vis, qt


def kabsch(A, Bp):
    """Rigid (R, t) minimising |R A + t - B| over (n, 3) point pairs."""
    ca, cb = A.mean(0), Bp.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def lift_points(uv, vis, depth, c2w, K):
    """Per visible track point, the world point from its depth: (T, N, 3) with nan where unavailable."""
    T, N = vis.shape
    Ki = np.linalg.inv(K)
    X = np.full((T, N, 3), np.nan, np.float32)
    for t in range(T):
        z = B.sample_depth(depth, t, uv[t])
        ok = vis[t] & np.isfinite(z) & (z > 0.05)
        P = (np.c_[uv[t], np.ones(N)] @ Ki.T) * z[:, None]
        X[t, ok] = (P @ c2w[t, :3, :3].T + c2w[t, :3, 3])[ok]
    return X


# ---------------------------------------------------------------- per video
class Video:
    def __init__(self, a, name, gdino):
        self.a, self.name, self.gdino = a, name, gdino
        self.src, self.dst = os.path.join(a.input_dir, name), os.path.join(a.output_dir, name)
        self.work = os.path.join(self.dst, '_work')
        self.cur = 'input'

    # -- state (sources / scale factors survive a restart)
    def load_state(self):
        p = os.path.join(self.work, 'state.json')
        self.state = json.load(open(p)) if os.path.exists(p) else dict(sources={}, scale_alignment={})

    def save_state(self):
        F.write_json(os.path.join(self.work, 'state.json'), self.state)

    def cache(self, key, fn):
        """npz cache of an intermediate result in _work (dict of arrays)."""
        p = os.path.join(self.work, f'{key}.npz')
        if os.path.exists(p):
            z = np.load(p)
            return {k: z[k] for k in z.files}
        r = fn()
        np.savez(p + '.tmp.npz', **r)
        os.replace(p + '.tmp.npz', p)
        return r

    # -- modality accessors (output copy wins: it is GT or a finished estimate)
    def K(self):
        return F.read_intrinsics(self.dst)

    def c2w(self):
        return F.read_extrinsics(self.dst)

    def depth(self):
        if getattr(self, '_depth', None) is None:
            self._depth = F.read_depth(self.dst)
        return self._depth

    def masks(self):
        if getattr(self, '_masks', None) is None:
            self._masks = F.read_masks(self.dst)
        return self._masks

    def monst3r(self):
        def run():
            log('  MonST3R global alignment')
            m = S.run_monst3r(self.frames, max_frames=self.a.monst3r_frames)
            free_gpu()
            return {f'mon_{k}': v for k, v in m.items()}
        return self.cache('monst3r', run)

    def tracks2d(self, depth=None):
        """2D tracks (GT if given, else CoTracker, cached). depth: use this instead of the output depth (sampling)."""
        if F.present(self.src, 'tracks'):
            t = F.read_tracks(self.src)
            return t['uv'], t['vis'], t['qt']

        def run():
            log('  CoTracker')
            uv, vis, qt = run_tracks(self.frames, self.depth() if depth is None else depth, self.masks(), self.segs, every=self.a.query_every,
                                     spacing=self.a.track_spacing or None, max_side=self.a.cotracker_max_side)
            return dict(uv=uv, vis=vis, qt=qt)
        r = self.cache('tracks2d', run)
        return r['uv'], r['vis'], r['qt']

    def run(self):
        a = self.a
        if os.path.exists(os.path.join(self.dst, 'DONE')):
            log(f'{self.name}: DONE, skipped')
            return True
        t0 = time.time()
        err, warn, _ = F.validate_video(self.src, deep=False)
        if err:
            raise ValueError('invalid input: ' + '; '.join(err))
        for w in warn:
            log(f'  input warning: {w}')
        os.makedirs(self.work, exist_ok=True)
        self.load_state()
        self.meta = F.read_meta(self.src)
        self.cur = 'rgb'
        if os.path.exists(os.path.join(self.src, 'video.mp4')):
            if not os.path.exists(os.path.join(self.dst, 'video.mp4')):
                F.link_or_copy(os.path.join(self.src, 'video.mp4'), os.path.join(self.dst, 'video.mp4'))
        elif not os.path.isdir(os.path.join(self.dst, 'rgb')):
            F.link_tree(os.path.join(self.src, 'rgb'), os.path.join(self.dst, 'rgb'))
        missing = [m for m in F.MODALITIES if not F.present(self.dst, m)]
        todo = [m for m in missing if not F.present(self.src, m)]
        tracks_incomplete = F.present(self.src, 'tracks') and not all(
            os.path.exists(os.path.join(self.src, 'tracks', f)) for f in ('tracks_3d.npy', 'query_frame.npy'))
        if todo or tracks_incomplete:
            load_specialists(a.scratch)
            self.frames = F.read_rgb(self.src)
            T, H, W = self.frames.shape[:3]
            assert T == self.meta['num_frames'] and [W, H] == list(self.meta['resolution']), 'RGB / meta mismatch'
            self.fps = float(self.meta['fps'])
            self.segs = segments(T, a.segment, a.overlap)
        log(f'{self.name}: {self.meta["num_frames"]} frames; GT {[m for m in F.MODALITIES if F.present(self.src, m)]}; '
            f'to estimate {todo}' + (' (+ tracks_3d / query_frame)' if tracks_incomplete else ''))
        for mod in F.MODALITIES:
            self.cur = mod
            if F.present(self.dst, mod):
                self.state['sources'].setdefault(mod, 'gt' if F.present(self.src, mod) else 'unknown')
                continue
            t = time.time()
            if F.present(self.src, mod) and not (mod == 'tracks' and tracks_incomplete):
                self.copy_gt(mod)
                self.state['sources'][mod] = 'gt'
            else:
                if getattr(self, f'estimate_{mod}')() is False:
                    continue  # deferred (intrinsics with extrinsics)
            self.save_state()
            log(f'  {mod}: {self.state["sources"][mod]} ({time.time() - t:.0f} s)')
        self.cur = 'finalize'
        for mod in F.MODALITIES:
            assert F.present(self.dst, mod), f'{mod} missing after processing'
        meta = dict(self.meta, sources={m: self.state['sources'][m] for m in F.MODALITIES},
                    scale_alignment=self.state['scale_alignment'],
                    stage_a=dict(version=VERSION, host=socket.gethostname(),
                                 finished=datetime.datetime.now().isoformat(timespec='seconds')))
        if os.path.exists(os.path.join(self.dst, 'masks', 'names.json')):
            meta['sources']['mask_names'] = self.state['sources'].get('mask_names', self.state['sources']['masks'])
        F.write_json(os.path.join(self.dst, 'meta.json'), meta)
        shutil.rmtree(self.work, ignore_errors=True)
        open(os.path.join(self.dst, 'DONE'), 'w').close()
        log(f'{self.name}: DONE in {time.time() - t0:.0f} s')
        return True

    def copy_gt(self, mod):
        if mod in ('depth', 'masks'):
            F.link_tree(os.path.join(self.src, mod), os.path.join(self.dst, mod))
        elif mod == 'tracks':
            F.link_tree(os.path.join(self.src, 'tracks'), os.path.join(self.dst, 'tracks'))
        else:
            F.link_or_copy(os.path.join(self.src, 'camera', f'{mod}.npy'), os.path.join(self.dst, 'camera', f'{mod}.npy'))

    # -- 1. intrinsics
    def estimate_intrinsics(self):
        if not F.present(self.src, 'extrinsics'):
            return False  # refined together with the estimated extrinsics (step 4)
        F.write_intrinsics(self.dst, self.monst3r()['mon_K'])
        self.state['sources']['intrinsics'] = 'monst3r'

    # -- 2. masks
    def estimate_masks(self):
        lab = sam2_segments(self.frames, self.fps, self.segs)
        names = None
        if not self.a.no_names:
            log('  Grounding DINO naming')
            names = name_instances(lab, self.frames, self.fps, self.gdino, [w.strip() for w in self.a.vocab.split(',')])
            self.state['sources']['mask_names'] = 'grounding_dino'
        F.write_masks(self.dst, lab, names)
        self._masks = lab
        self.state['sources']['masks'] = 'sam2.1_auto'

    # -- 3. depth
    def estimate_depth(self):
        log('  Video Depth Anything (metric)')
        z = S.run_vda(self.frames, self.fps)
        free_gpu()
        src = 'video_depth_anything_metric_vitl'
        if F.present(self.src, 'extrinsics'):
            uv, vis, _ = self.tracks2d(depth=z)  # depth only steers the query sampling (scale-free edge test)
            z, info = align_depth_to_cameras(z, uv, vis, F.read_extrinsics(self.src), self.K())
            self.state['scale_alignment']['depth'] = dict(to='gt extrinsics', **info)
            log(f'  depth aligned to the GT cameras: {info}')
        F.write_depth(self.dst, z)
        self._depth = None  # re-read: the stored depth is quantised to mm and capped at 65.535 m
        self.state['sources']['depth'] = src

    # -- 4. extrinsics
    def estimate_extrinsics(self):
        a = self.a
        depth = self.depth()
        T = len(depth)
        gt_K = F.present(self.src, 'intrinsics')
        uv, vis, _ = self.tracks2d()
        mon = self.monst3r() if a.pose_method == 'monst3r' or not gt_K else None
        if a.pose_method == 'vo':
            K = self.K() if gt_K else mon['mon_K']
            c2w = B.vo_cameras(dict(ct_uv=uv, ct_vis=vis, det_frame=np.zeros(0, int)), depth, K)
            src, info = 'pnp_vo', dict(factor=1.0, to='metric depth (PnP on depth-lifted tracks)')
        else:
            K0 = self.K() if gt_K else mon['mon_K']
            raw = dict(mon, mon_K=K0)
            c2w, _, sc = B.cameras(raw, depth)
            if not np.isfinite(sc):
                raise RuntimeError('MonST3R / depth scale undetermined (no confident static pixels)')
            c2w, K = B.bundle_adjust(uv, vis, depth, c2w, K0)
            if gt_K:
                K = K0
            src = 'monst3r+bundle_adjustment'
            info = dict(factor=round(float(sc), 5), to='gt depth' if F.present(self.src, 'depth') else 'estimated depth',
                        method='median depth ratio on confident static MonST3R pixels')
        if F.present(self.src, 'tracks') and os.path.exists(os.path.join(self.src, 'tracks', 'tracks_3d.npy')):
            # GT 3D tracks fix the world frame: move the estimated cameras into it (rigid, from lifted track points)
            X = F.read_tracks(self.src)['X']
            Xe = lift_points(uv, vis, depth, c2w, K)
            ok = np.isfinite(Xe[..., 0]) & np.isfinite(X[..., 0])
            A, Bp = Xe[ok].astype(np.float64), X[ok].astype(np.float64)
            if len(A) >= 100:
                for _ in range(3):  # trimmed: drop the worst 20% (moving points, depth errors)
                    R, t = kabsch(A, Bp)
                    r = np.linalg.norm(A @ R.T + t - Bp, axis=1)
                    keep = r <= np.percentile(r, 80)
                    A, Bp = A[keep], Bp[keep]
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R, t
                c2w = M @ c2w
                info['world_frame'] = f'aligned to GT tracks_3d (rigid), median residual {np.median(r):.3f} m'
        if not gt_K:  # before the extrinsics: a present extrinsics.npy implies the deferred intrinsics are written
            F.write_intrinsics(self.dst, K)
            self.state['sources']['intrinsics'] = 'monst3r' + ('+bundle_adjustment' if a.pose_method != 'vo' else '')
            self.save_state()
        F.write_extrinsics(self.dst, c2w)
        self.state['sources']['extrinsics'] = src
        self.state['scale_alignment']['extrinsics'] = info

    # -- 5. tracks
    def estimate_tracks(self):
        depth, c2w, K, masks = self.depth(), self.c2w(), self.K(), self.masks()
        uv, vis, qt = self.tracks2d()
        gt = F.present(self.src, 'tracks')
        if qt is None:
            qt = np.argmax(vis, 0)
        fake = dict(mon_crop=np.zeros(5), mon_sel=np.zeros(0, int), mon_dyn=np.zeros((0, 1, 1), bool))
        log(f'  lifting {uv.shape[1]} tracks')
        X, dyn, keep, _, _ = B.lift_tracks(uv, vis, depth, c2w, K, fake, masks, qt)
        if gt:
            src_t = F.read_tracks(self.src)
            if src_t['X'] is not None:
                X = src_t['X']
            else:
                X = np.where(keep[None, :, None], X, np.nan).astype(np.float32)
            F.write_tracks(self.dst, uv, vis, qt, X)
            self.state['sources']['tracks'] = 'gt' + ('' if src_t['X'] is not None else ' (tracks_3d: lifted)') + \
                ('' if src_t['qt'] is not None else ' (query_frame: first visible frame)')
        else:
            F.write_tracks(self.dst, uv[:, keep], vis[:, keep], qt[keep], X[:, keep])
            self.state['sources']['tracks'] = 'cotracker3_online+lifted'
        self.state['tracks'] = dict(n=int(keep.sum() if not gt else len(keep)), n_dynamic=int((dyn & keep).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--videos', default='', help='comma list of video names (default: all)')
    ap.add_argument('--shard', default='0/1', help='i/n: process every n-th video starting at i (parallel jobs)')
    ap.add_argument('--segment', type=int, default=300, help='SAM2 / CoTracker segment length (frames)')
    ap.add_argument('--overlap', type=int, default=16, help='segment overlap (frames)')
    ap.add_argument('--query_every', type=int, default=12, help='CoTracker query frame interval')
    ap.add_argument('--track_spacing', type=int, default=0, help='query grid spacing px (0: 12 px per 960 px)')
    ap.add_argument('--cotracker_max_side', type=int, default=960, help='CoTracker input resolution (long side)')
    ap.add_argument('--monst3r_frames', type=int, default=50, help='frames used by MonST3R global alignment')
    ap.add_argument('--pose_method', choices=['monst3r', 'vo'], default='monst3r',
                    help='monst3r: MonST3R + bundle adjustment; vo: PnP visual odometry on metric depth (driving)')
    ap.add_argument('--vocab', default=VOCAB, help='Grounding DINO vocabulary for mask names (comma separated)')
    ap.add_argument('--no_names', action='store_true', help='do not name estimated masks (skips Grounding DINO)')
    ap.add_argument('--gdino_python', default=QWEN_PY, help='python of an env with transformers + torch >= 2.4')
    ap.add_argument('--scratch', default=os.environ.get('PGT_SCRATCH', f'/tmp/{os.environ.get("USER", "u")}_stage_a'),
                    help='local scratch dir (frames for SAM2 / MonST3R)')
    a = ap.parse_args()
    a.input_dir, a.output_dir = os.path.abspath(a.input_dir), os.path.abspath(a.output_dir)
    if a.output_dir == a.input_dir or a.output_dir.startswith(a.input_dir + os.sep):
        sys.exit('output_dir must not be inside input_dir')
    os.makedirs(a.output_dir, exist_ok=True)
    names = F.video_dirs(a.input_dir)
    if a.videos:
        want = set(a.videos.split(','))
        names = [n for n in names if n in want]
    i, n = map(int, a.shard.split('/'))
    names = names[i::n]
    log(f'stage_a: {len(names)} videos (shard {a.shard}) {a.input_dir} -> {a.output_dir}')
    gdino = GDino(a.gdino_python, a.scratch)
    ok = fail = 0
    try:
        for name in names:
            v = Video(a, name, gdino)
            try:
                v.run()
                ok += 1
            except Exception as e:
                fail += 1
                tb = traceback.format_exc()
                log(f'{name}: FAILED in {v.cur}: {e}\n{tb}')
                rec = dict(video=name, modality=v.cur, error=f'{type(e).__name__}: {e}', traceback=tb,
                           host=socket.gethostname(), time=datetime.datetime.now().isoformat(timespec='seconds'))
                with open(os.path.join(a.output_dir, 'failures.jsonl'), 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                if S is not None:
                    free_gpu()
    finally:
        gdino.close()
    log(f'stage_a finished: {ok} ok, {fail} failed')
    return 3 if fail else 0  # 3: ran to the end, failures are in failures.jsonl (not retried by condor)


if __name__ == '__main__':
    sys.exit(main())
