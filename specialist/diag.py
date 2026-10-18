# Per-component check of a pseudo-GT PO scene against PointOdyssey GT (the clip's source frames).
#   python diag.py T3_001 [T1_001 ...]      -> prints metrics, appends to <ROOT>/eval/diag.json
#   depth:   VDA z vs GT z (scale = median ratio, AbsRel after scaling, delta<1.25)
#   camera:  focal vs GT; camera centres after similarity alignment (ATE, relative to path length); rotation error
#   tracks:  every pseudo grid track vs the nearest GT track visible at its query frame (<= 4 px): px error over
#            the clip, visibility agreement; static/dynamic label vs GT (world motion > 5 cm)
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE (see build.py)
import io, json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from build import CLIPS, PO_ROOT, ROOT
from blk import PO, read


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def diag(name):
    c = next(c for c in json.load(open(CLIPS)) if c['name'] == name)
    st, en = c['start'], c['end']
    gd = f"{PO}/{c['scene']}"
    g = np.load(f'{gd}/anno.npz')
    pd = f'{PO_ROOT}/pseudo/{name}'
    p = np.load(f'{pd}/anno.npz')
    pr = np.load(f'{pd}/probes.npz')
    T = en - st + 1
    out = dict(clip=name, scene=c['scene'])
    # depth
    rs, ab, d1 = [], [], []
    for t in range(0, T, 12):
        zg = np.array(Image.open(io.BytesIO(read(gd, 'depths', f'depth_{t + st:05d}.png')))).astype(np.float32) * (1000 / 65535)
        zp = np.array(Image.open(io.BytesIO(read(pd, 'depths', f'depth_{t:05d}.png')))).astype(np.float32) * (1000 / 65535)
        ok = (zg > 0.05) & (zg < 100) & (zp > 0.05)
        rs.append(np.median(zp[ok] / zg[ok]))
    s_d = float(np.median(rs))
    for t in range(0, T, 12):
        zg = np.array(Image.open(io.BytesIO(read(gd, 'depths', f'depth_{t + st:05d}.png')))).astype(np.float32) * (1000 / 65535)
        zp = np.array(Image.open(io.BytesIO(read(pd, 'depths', f'depth_{t:05d}.png')))).astype(np.float32) * (1000 / 65535)
        ok = (zg > 0.05) & (zg < 100) & (zp > 0.05)
        q = zp[ok] / s_d
        ab.append(np.mean(np.abs(q - zg[ok]) / zg[ok]))
        d1.append(np.mean(np.maximum(q / zg[ok], zg[ok] / q) < 1.25))
    out['depth'] = dict(scale_pseudo_over_gt=s_d, scale_spread=float(np.std(rs) / np.mean(rs)), absrel=float(np.mean(ab)),
                        delta1=float(np.mean(d1)))
    # camera
    Eg, Ep = g['extrinsics'][st:en + 1].astype(np.float64), p['extrinsics'].astype(np.float64)
    Cg = np.array([-e[:3, :3].T @ e[:3, 3] for e in Eg])
    Cp = np.array([-e[:3, :3].T @ e[:3, 3] for e in Ep])
    s, R, t = umeyama(Cp, Cg)
    ate = np.linalg.norm((s * Cp @ R.T + t) - Cg, axis=1)
    path = np.linalg.norm(np.diff(Cg, axis=0), axis=1).sum()
    # relative rotation error between consecutive-second frames
    rerr = []
    for k in range(0, T - 24, 6):
        Rg = Eg[k + 24, :3, :3] @ Eg[k, :3, :3].T
        Rp = Ep[k + 24, :3, :3] @ Ep[k, :3, :3].T
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Rp) - 1) / 2, -1, 1))))
    rot_g = [np.degrees(np.arccos(np.clip((np.trace(Eg[k + 24, :3, :3] @ Eg[k, :3, :3].T) - 1) / 2, -1, 1)))
             for k in range(0, T - 24, 6)]
    out['camera'] = dict(f_pseudo=float(p['intrinsics'][0, 0, 0]), f_gt=float(g['intrinsics'][st, 0, 0]),
                         traj_scale_gt_over_pseudo=float(s), ate_m=float(ate.mean()), gt_path_m=float(path),
                         gt_cam_speed=float(path / (T / 24)), rel_rot_err_deg_1s=float(np.mean(rerr)),
                         gt_rot_deg_1s=float(np.mean(rot_g)))
    # tracks
    uvg, visg = g['trajs_2d'][st:en + 1], (g['visibs'] & g['valids'])[st:en + 1]
    Xg = g['trajs_3d'][st:en + 1]
    uvp, visp = p['trajs_2d'], p['visibs']
    qt = np.argmax(visp, 0)  # first visible frame ~ query frame
    from scipy.spatial import cKDTree
    errs, vagree, dyn_p, dyn_g, rel_err = [], [], [], [], []
    Pp = np.einsum('tij,tnj->tni', Ep[:, :3, :3], p['trajs_3d']) + Ep[:, None, :3, 3]
    Pg_all = np.einsum('tij,tnj->tni', Eg[:, :3, :3], Xg) + Eg[:, None, :3, 3]
    for t in np.unique(qt):
        jp = np.nonzero(qt == t)[0]
        cand = np.nonzero(visg[t])[0]
        if not len(cand):
            continue
        dd, k = cKDTree(uvg[t, cand]).query(uvp[t, jp])
        okm = dd <= 4
        jp, jg = jp[okm], cand[k[okm]]
        for a, b in zip(jp, jg):
            vv = visg[:, b]
            e = np.linalg.norm(uvp[:, a] - uvg[:, b], axis=1)
            errs.append(np.median(e[vv]) if vv.any() else np.nan)
            vagree.append((visp[:, a] == vv).mean())
            dyn_g.append(np.linalg.norm(Xg[:, b] - Xg[:1, b], axis=1).max() > 0.05)
            rp, rg = np.linalg.norm(Pp[:, a], axis=1), np.linalg.norm(Pg_all[:, b], axis=1)
            both = vv & visp[:, a]
            if both.sum() > 5:
                cp, cg = rp[both] / rp[both][0], rg[both] / rg[both][0]
                rel_err.append(np.abs(cp - cg).mean())
        dyn_p += list(pr['dyn_grid'][jp])
    errs = np.array(errs)
    dyn_p, dyn_g = np.array(dyn_p, bool), np.array(dyn_g, bool)
    out['tracks'] = dict(n_pseudo=int(uvp.shape[1]), n_gt=int(uvg.shape[1]), n_matched=len(errs),
                         px_err_median=float(np.nanmedian(errs)), px_err_p90=float(np.nanpercentile(errs, 90)),
                         vis_agree=float(np.mean(vagree)), relcurve_mae=float(np.mean(rel_err)),
                         dyn_gt_frac=float(dyn_g.mean()), dyn_pseudo_frac=float(dyn_p.mean()),
                         dyn_recall=float((dyn_p & dyn_g).sum() / max(dyn_g.sum(), 1)),
                         dyn_precision=float((dyn_p & dyn_g).sum() / max(dyn_p.sum(), 1)))
    out['masks'] = dict(n_pseudo=int(max(np.array(Image.open(io.BytesIO(read(pd, 'masks', f'mask_{t:05d}.png')))).max()
                                         for t in range(0, T, 24))),
                        n_gt_frame0=int(len(np.unique(np.array(Image.open(io.BytesIO(read(gd, 'masks', f'mask_{st:05d}.png'))))))))
    return out


if __name__ == '__main__':
    res = []
    for n in sys.argv[1:]:
        r = diag(n)
        res.append(r)
        print(json.dumps(r, indent=1), flush=True)
    os.makedirs(f'{ROOT}/eval', exist_ok=True)
    path = f'{ROOT}/eval/diag.json'
    old = json.load(open(path)) if os.path.exists(path) else []
    old = [o for o in old if o['clip'] not in sys.argv[1:]] + res
    json.dump(old, open(path, 'w'), indent=1)
