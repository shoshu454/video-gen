# Validate a standard-format folder (README_FORMAT.md) before running stage_a.py.
#   python check_inputs.py <dir> [--videos a,b] [--quick]
# Per video: required files, shapes, dtypes, frame counts, depth plausibility (median in metres), rigid extrinsics,
# and, when depth + intrinsics + extrinsics exist, a static-point check: background pixels of frame t are unprojected
# with depth, moved to frame t + k with the extrinsics and compared with frame t + k (depth, colour and, with tracks,
# the tracked 2D position; its 25th percentile is reported because moving points also fail it). Colour error with
# the warp (reproj_photo) vs. without it (photo_no_motion) is informational. The same check with the extrinsics inverted tells c2w from w2c. With tracks_3d, its
# projection is compared with tracks_2d. Exit code 1 if any video has errors.
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stdfmt as F


def edges(z, ratio=1.1):
    from scipy.ndimage import maximum_filter, minimum_filter
    mx = maximum_filter(np.where(z > 0, z, 0), 5)
    mn = minimum_filter(np.where(z > 0, z, np.inf), 5)
    return ~(z > 0) | ~(mx <= ratio * mn)


def reproject(z0, c2w0, c2w1, K, u, v):
    """Pixels (u, v) of frame 0 with depth z0 -> pixel coords and depth in frame 1."""
    P = (np.c_[u, v, np.ones(len(u))] @ np.linalg.inv(K).T) * z0[:, None]
    Xw = P @ c2w0[:3, :3].T + c2w0[:3, 3]
    E1 = np.linalg.inv(c2w1)
    Q = Xw @ E1[:3, :3].T + E1[:3, 3]
    q = Q @ K.T
    return q[:, 0] / np.maximum(q[:, 2], 1e-9), q[:, 1] / np.maximum(q[:, 2], 1e-9), Q[:, 2]


def pair_metrics(rgb, depth, masks, c2w, K, tracks, t0, t1, n=6000, seed=0):
    H, W = depth.shape[1:]
    z0 = depth[0]
    ok = ~edges(z0)
    use_bg = masks is not None and (masks[0] == 0).mean() >= 0.2  # id 0 is a usable background
    if use_bg:
        ok &= masks[0] == 0
    v, u = np.nonzero(ok)
    if len(u) < 50:
        return None
    j = np.random.default_rng(seed).choice(len(u), min(n, len(u)), replace=False)
    u, v = u[j], v[j]
    out = {}
    for hyp, cc in (('c2w', c2w), ('w2c', np.linalg.inv(c2w))):
        u1, v1, zp = reproject(z0[v, u], cc[t0], cc[t1], K, u.astype(float), v.astype(float))
        ui, vi = np.round(u1).astype(int), np.round(v1).astype(int)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (zp > 1e-3)
        if inb.sum() < 50:
            out[hyp] = dict(inside=float(inb.mean()))
            continue
        z1 = depth[1][vi[inb], ui[inb]]
        good = z1 > 0
        rel = np.abs(zp[inb][good] - z1[good]) / z1[good]
        photo = np.abs(rgb[0][v[inb], u[inb]].astype(float) - rgb[1][vi[inb], ui[inb]].astype(float)).mean()
        out[hyp] = dict(inside=float(inb.mean()), depth_relerr=float(np.median(rel)) if good.any() else np.nan,
                        photo=float(photo))
    out['photo_no_motion'] = float(np.abs(rgb[0][v, u].astype(float) - rgb[1][v, u].astype(float)).mean())
    if tracks is not None:
        uv, vis = tracks['uv'], tracks['vis']
        s = vis[t0] & vis[t1]
        if s.sum() >= 10:
            a = uv[t0, s]
            ai = np.clip(np.round(a).astype(int), 0, [W - 1, H - 1])
            za = z0[ai[:, 1], ai[:, 0]]
            bg = (za > 0) & ~edges(z0)[ai[:, 1], ai[:, 0]]
            if tracks.get('X') is not None:  # static per tracks_3d (selection only; the check uses 2D + depth + poses)
                st = np.linalg.norm(tracks['X'][t1, s] - tracks['X'][t0, s], axis=1) < 0.01
                if st.sum() >= 10:
                    bg &= st
                    out['track_static_from_3d'] = True
            elif use_bg:
                bg &= masks[0][ai[:, 1], ai[:, 0]] == 0
            if bg.sum() >= 10:
                u1, v1, _ = reproject(za[bg], c2w[t0], c2w[t1], K, a[bg, 0], a[bg, 1])
                e = np.hypot(u1 - uv[t1, s][bg, 0], v1 - uv[t1, s][bg, 1])
                out['track_reproj_px'] = float(np.median(e))
                out['track_reproj_px_p25'] = float(np.percentile(e, 25))  # robust to moving points
    return out


def static_check(d, m, n_pairs=5):
    """Returns (warnings, info)."""
    warn, info = [], {}
    T = int(m['num_frames'])
    K, c2w = F.read_intrinsics(d), F.read_extrinsics(d)
    C = c2w[:, :3, 3]
    gap = max(1, int(round(float(m['fps']) / 3)))
    if T <= gap:
        return warn, info
    tracks = F.read_tracks(d) if F.present(d, 'tracks') else None
    starts = np.linspace(0, T - 1 - gap, min(n_pairs, T - gap)).round().astype(int)
    res = []
    for t0 in starts:
        t1 = t0 + gap
        rgb = F.read_rgb(d, [t0, t1])
        depth = F.read_depth(d, [t0, t1])
        masks = F.read_masks(d, [t0, t1]) if F.present(d, 'masks') else None
        r = pair_metrics(rgb, depth, masks, c2w, K, tracks, t0, t1)
        if r is not None:
            res.append(r)
    if not res:
        warn.append('static-point check: not enough valid background pixels')
        return warn, info
    med = lambda key, hyp=None: float(np.nanmedian([(r[hyp] if hyp else r).get(key, np.nan) for r in res
                                                    if (hyp is None or hyp in r)]))
    dc, dw = med('depth_relerr', 'c2w'), med('depth_relerr', 'w2c')
    pc, pw, p0 = med('photo', 'c2w'), med('photo', 'w2c'), med('photo_no_motion')
    info.update(static_gap_frames=gap, reproj_depth_relerr=round(dc, 4), reproj_photo=round(pc, 2),
                photo_no_motion=round(p0, 2), reproj_depth_relerr_if_w2c=round(dw, 4))
    if any('track_reproj_px' in r for r in res):
        info['track_reproj_px'] = round(med('track_reproj_px'), 2)
        info['track_reproj_px_p25'] = round(med('track_reproj_px_p25'), 2)
        info['track_static_from'] = 'tracks_3d' if any(r.get('track_static_from_3d') for r in res) else 'percentile'
    moved = float(np.median(np.linalg.norm(C[starts + gap] - C[starts], axis=1)))
    info['baseline_m'] = round(moved, 4)
    if np.isfinite(dw) and (not np.isfinite(dc) or dw < 0.5 * dc) and dw < 0.1:
        warn.append(f'static-point check: extrinsics look like WORLD-TO-CAMERA (depth reprojection error {dc:.1%} as '
                    f'c2w, {dw:.1%} inverted): store camera-to-world')
    elif np.isfinite(dc) and dc > 0.10:
        warn.append(f'static-point check: depth reprojection error {dc:.1%} (expected < 5%): check depth units, '
                    'z vs ray distance, OpenGL vs OpenCV axes, frame alignment')
    if info.get('track_reproj_px_p25', 0) > 3:
        warn.append(f'static-point check: the best quarter of the tracks reprojects {info["track_reproj_px_p25"]:.1f} '
                    'px or more from its tracked position (expected < 3 px even with moving objects)')
    return warn, info


def tracks3d_check(d):
    """tracks_3d projected with the extrinsics / intrinsics vs tracks_2d at visible points (median px)."""
    t = F.read_tracks(d)
    if t['X'] is None or not t['vis'].any():
        return [], {}
    K, c2w = F.read_intrinsics(d), F.read_extrinsics(d)
    E = np.linalg.inv(c2w)
    T = len(c2w)
    sel = np.linspace(0, T - 1, min(T, 20)).round().astype(int)
    errs, front = [], []
    for f in sel:
        m = t['vis'][f] & np.isfinite(t['X'][f, :, 0])
        if not m.any():
            continue
        P = t['X'][f, m] @ E[f, :3, :3].T + E[f, :3, 3]
        q = P @ K.T
        ok = q[:, 2] > 1e-6
        front.append(ok.mean())
        if ok.any():
            e = np.hypot(q[ok, 0] / q[ok, 2] - t['uv'][f, m][ok, 0], q[ok, 1] / q[ok, 2] - t['uv'][f, m][ok, 1])
            errs.append(np.median(e))
    if front and np.mean(front) < 0.9:
        return [f'{1 - np.mean(front):.0%} of the visible tracks_3d points are behind the camera: OpenGL axes '
                '(z backward) or world-to-camera extrinsics?'], dict(tracks3d_in_front=round(float(np.mean(front)), 3))
    if not errs:
        return [], {}
    e = float(np.median(errs))
    w = [f'tracks_3d projects {e:.1f} px from tracks_2d (expected < 2 px): world frame / extrinsics mismatch?'] if e > 2 else []
    return w, dict(tracks3d_reproj_px=round(e, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--videos', default='')
    ap.add_argument('--quick', action='store_true', help='only read the first / last depth and mask PNG')
    a = ap.parse_args()
    names = F.video_dirs(a.dir)
    if a.videos:
        names = [n for n in names if n in set(a.videos.split(','))]
    if not names:
        sys.exit(f'no video folders with meta.json in {a.dir}')
    bad = 0
    for n in names:
        d = os.path.join(a.dir, n)
        err, warn, info = F.validate_video(d, deep=not a.quick)
        if not err:
            m = F.read_meta(d)
            have = [x for x in F.MODALITIES if F.present(d, x)]
            info['modalities'] = ','.join(have) or 'rgb only'
            try:
                if all(F.present(d, x) for x in ('depth', 'intrinsics', 'extrinsics')):
                    w, i = static_check(d, m)
                    warn += w
                    info.update(i)
                if all(F.present(d, x) for x in ('tracks', 'intrinsics', 'extrinsics')):
                    w, i = tracks3d_check(d)
                    warn += w
                    info.update(i)
            except Exception as e:
                err.append(f'consistency check crashed: {type(e).__name__}: {e}')
        status = 'ERROR' if err else ('WARN' if warn else 'OK')
        bad += bool(err)
        print(f'[{status}] {n}  ' + '  '.join(f'{k}={v}' for k, v in info.items()))
        for e in err:
            print(f'    error: {e}')
        for w in warn:
            print(f'    warning: {w}')
    print(f'{len(names)} videos, {bad} with errors')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
