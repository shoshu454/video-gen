# Check pseudo-GT against dataset GT on the clips of clips.json (qwen_video env, CPU).
#   A) re-answer: each clip's GT item is re-labelled from pseudo-GT only (its marked points were tracked as CoTracker
#      probes from their t0 pixels; T6: lifted with pseudo depth and moved with the pseudo ego pose) using the
#      walkthroughs' own label rules; compared with the GT answer, plus probe track / visibility / distance errors.
#   B) generated items: every item rendered by run_pseudo.py is mapped back to GT (each pseudo point -> nearest GT
#      track visible at t0; T6 -> nearest GT LiDAR return, GT OXTS events) and re-labelled with GT.
# Writes <ROOT>/eval/{A,B}.json and summary.md.
#   python eval_gt.py [A|B|C|both]
#   C) every selection candidate (not only the rendered items) re-labelled with GT, for larger samples.
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE: THP compaction stalls make big allocations take minutes on vision-c22
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [HERE, SRC, f'{SRC}/prod']
from build import CLIPS, KITTI_ROOT, PO_ROOT, RAW, ROOT, VELO_TO_CAM, sample_depth
import blk
import common
import run

GT_PO = '/shared/rsaas/joeyw3/data/PointOdyssey/packed'  # not blk.PO: run_pseudo may have redirected it
GT_KR = '/shared/rsaas/joeyw3/data/kitti_raw'
EVAL = f'{ROOT}/eval'
ITEMS = f'{ROOT}/qa'
NS = {}


def ns(tpl):
    if tpl not in NS:
        NS[tpl], _ = run.load_ns(tpl, run.base_argv(tpl) + (['--drives', 'none'] if tpl == 'T6' else ['--scenes', 'none']))
        if tpl == 'T6':
            NS[tpl]['tracklets'] = _tracklets_or_none(NS[tpl]['tracklets'])
    return NS[tpl]


_gt_cache = {}


def gt_po(scene):
    """GT PointOdyssey scene through the T3 walkthrough loader (r, onscreen, per-frame K, depth, masks)."""
    if scene not in _gt_cache:
        _gt_cache.clear()
        n3 = ns('T3')
        common.PO, n3['PO'] = GT_PO, GT_PO
        d = n3['load'](scene)
        d['thr'] = n3['OVL'] * d['W']
        _gt_cache[scene] = d
    return _gt_cache[scene]


def gt_kitti(drive):
    key = 'kitti/' + drive
    if key not in _gt_cache:
        n6 = ns('T6')
        n6['KR'] = GT_KR
        _gt_cache[key] = n6['load'](drive)
    return _gt_cache[key]


def _tracklets_or_none(load_trk):
    """Most drives outside 2011_09_26 have no tracklet_labels.xml: load them without boxes (no object exclusion)."""
    def f(path):
        return load_trk(path) if os.path.exists(path) else []
    return f


def pseudo_kitti(drive_id):
    n6 = ns('T6')
    n6['KR'] = KITTI_ROOT
    d = n6['load'](drive_id)
    n6['KR'] = GT_KR
    return d


def in_bounds(uv, W, H):
    return (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)


def po_label(tpl, uv, vis, r, onscreen, fb, t0, tn, thr, letters):
    """Walkthrough label rules on per-point arrays (T, P, ...) for points 0..P-1. Returns (answer, detail)."""
    if tpl == 'T1':  # the template's margin M = 0.10: a smaller change has no answer
        rel = float((r[tn, 0] - r[t0, 0]) / r[t0, 0])
        return ('ambiguous' if abs(rel) < 0.10 else 'Farther' if rel > 0 else 'Closer'), dict(rel_change=rel)
    if tpl == 'T3':
        n3 = ns('T3')
        d = dict(uv=uv, vis=vis, r=r, onscreen=onscreen, thr=thr)
        dist, close, a_front, b_front = n3['pair_trace'](d, fb, 0, 1)
        lab, kind = n3['label_of'](close, a_front, b_front)
        return (lab or 'ambiguous'), dict(kind=kind, n_close=int(close.sum()), min_dist_px=float(dist.min()))
    if tpl == 'T4':
        order = sorted(letters, key=lambda L: r[tn, letters.index(L)])
        mn = min(abs(r[tn, a] - r[tn, b]) / min(r[tn, a], r[tn, b]) for a in range(3) for b in range(a + 1, 3))
        return (order if mn >= 0.10 else 'ambiguous'), dict(min_margin_tn=float(mn), order=order)
    if tpl == 'T5':
        n5 = ns('T5')
        R = r[fb]
        state, flip_at, init_ok, sm = n5['flip_trace'](R[:, :1], R[:, 1:])
        if not init_ok[0]:
            return 'ambiguous', dict(n_flips=None)
        n = int(flip_at.sum())
        return n5['flip_label'](n), dict(n_flips=n)


def track_errors(uv_p, vis_p, r_p, uv_g, vis_g, r_g):
    """Per point over baked frames: px error where GT visible, visibility agreement, error of the relative distance
    curve r(t)/r(t0), and the metric scale ratio r_pseudo/r_gt at t0."""
    e = np.linalg.norm(uv_p - uv_g, axis=-1)
    ev = e[vis_g] if vis_g.any() else e
    cp, cg = r_p / r_p[0], r_g / r_g[0]
    return dict(px_med=float(np.median(ev)), px_p90=float(np.percentile(ev, 90)), px_t0=float(e[0]),
                vis_agree=float((vis_p == vis_g).mean()), relcurve_mae=float(np.abs(cp - cg).mean()),
                scale_t0=float(r_p[0] / r_g[0]))


def answers_equal(tpl, a, b):
    return list(a) == list(b) if tpl == 'T4' else a == b


# ---------------------------------------------------------------- A: re-answer GT items from pseudo-GT
def eval_A(clips):
    out = []
    for c in clips:
        tpl = c['template']
        if not c.get('gt_item'):  # extra clips from GT candidates: no GT item to re-answer (B and C only)
            continue
        try:
            item = json.load(open(f"/shared/rsaas/joeyw3/pvideo/{tpl}/{c['gt_item']}.json"))
            letters = list(item['points'])
            fb_src = np.array(item['baked_frames'])
            fb = fb_src - c['start']
            t0, tn = c['t0'], c['tn']
            if tpl == 'T6':
                res = eval_A_kitti(c, item, letters, fb)
            else:
                pd = f"{PO_ROOT}/pseudo/{c['name']}"
                if not os.path.exists(f'{pd}/probes.npz'):
                    continue
                pr = np.load(f'{pd}/probes.npz')
                info = json.load(open(f'{pd}/pseudo_info.json'))
                W, H = 960, 540
                uv, vis, P = pr['uv'], pr['vis'], pr['P']
                r = np.linalg.norm(P, axis=-1)
                ons = pr['valid'] & (P[..., 2] > 0.05) & in_bounds(uv, W, H)
                ans, det = po_label(tpl, uv, vis, r, ons, fb, t0, tn, 0.01 * W, letters)
                gt_ans = item['ranking_tn'] if tpl == 'T4' else item['answer']
                g = gt_po(c['scene'])
                ji = [item['track_indices'][L] for L in letters]
                errs = {L: track_errors(uv[fb, k], vis[fb, k], r[fb, k], g['uv'][fb_src, j], g['vis'][fb_src, j],
                                        g['r'][fb_src, j]) for k, (L, j) in enumerate(zip(letters, ji))}
                dyn_gt = [bool(np.linalg.norm(g['X'][:, j] - g['X'][:1, j], axis=-1).max() > 0.05) for j in ji]
                res = dict(pseudo_answer=ans, gt_answer=gt_ans, correct=answers_equal(tpl, ans, gt_ans), detail=det,
                           point_errors=errs, dyn_pseudo=[bool(x) for x in pr['dyn']], dyn_gt=dyn_gt,
                           metric_scale=info['metric_scale'])
            out.append(dict(clip=c['name'], template=tpl, video=c.get('scene', c.get('drive')), **res))
            print('A', c['name'], res['pseudo_answer'], 'vs GT', res['gt_answer'], 'OK' if res['correct'] else 'WRONG', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out.append(dict(clip=c['name'], template=tpl, error=repr(e)[:300]))
    return out


def lift_kitti_points(drive_id, name, t0, uvs):
    """Pseudo T6 points: pseudo metric depth at the pixels on frame t0 -> velodyne coordinates at t0."""
    raw = np.load(f'{RAW}/{name}.npz')
    depth = raw['depth']
    info = json.load(open(glob.glob(f'{KITTI_ROOT}/{drive_id[:10]}/{drive_id}_sync/pseudo_info.json')[0]))
    K = np.array(info['K'])
    uvs = np.asarray(uvs, float)
    z = sample_depth(depth, t0, uvs).astype(float)
    Xc = (np.c_[uvs, np.ones(len(uvs))] @ np.linalg.inv(K).T) * z[:, None]
    return Xc @ VELO_TO_CAM


def eval_A_kitti(c, item, letters, fb):
    n6 = ns('T6')
    dp = pseudo_kitti(c['drive_id'])
    t0, te = c['t0'], c['te']
    Xv = lift_kitti_points(c['drive_id'], c['name'], t0, [item['points'][L]['uv_t0'] for L in letters])
    r = np.array([np.linalg.norm(n6['xform'](n6['velo_t0_to_cam'](dp, t0, int(t)), Xv), axis=1) for t in fb])
    ke = int(item['event_baked_frame']) - 1
    ans = letters[int(np.argmin(r[ke]))]
    rg = np.array([item['points'][L]['r'] for L in letters]).T  # GT distances on the baked frames
    errs = {L: dict(relcurve_mae=float(np.abs(r[:, k] / r[0, k] - rg[:, k] / rg[0, k]).mean()),
                    scale_t0=float(r[0, k] / rg[0, k]), r_te_pseudo=float(r[ke, k]), r_te_gt=float(rg[ke, k]))
            for k, L in enumerate(letters)}
    return dict(pseudo_answer=ans, gt_answer=item['answer'], correct=ans == item['answer'], point_errors=errs,
                detail=dict(event=c['event']))


# ---------------------------------------------------------------- B: generated items checked with GT
def eval_B(clips, tpls=None, prev=()):
    """tpls: only these templates; prev: earlier results, reused for items they cover without an error."""
    by_vid = {f"pseudo/{c['name']}": c for c in clips if c['src'] == 'po'}
    by_vid.update({c['drive_id']: c for c in clips if c['src'] == 'kitti'})
    old = {r['item']: r for r in prev if 'error' not in r}
    out = []
    for f in sorted(glob.glob(f'{ITEMS}/T*/T*_*.json')):
        item = json.load(open(f))
        tpl = item['template']
        c = by_vid.get(item['video_id'])
        if c is None or (tpls and tpl not in tpls):
            continue
        if item['item_id'] in old and old[item['item_id']].get('clip') == c['name']:
            out.append(old[item['item_id']])
            continue
        try:
            res = eval_B_kitti(c, item) if tpl == 'T6' else eval_B_po(c, item)
            out.append(dict(item=item['item_id'], template=tpl, clip=c['name'], video=c.get('scene', c.get('drive')),
                            pseudo_answer=item.get('ranking_tn', item['answer']) if tpl == 'T4' else item['answer'],
                            overlay=f"{ITEMS}/{tpl}/overlay/{item['item_id']}_overlay.mp4", **res))
            o = out[-1]
            print('B', item['item_id'], c['name'], o['pseudo_answer'], 'vs GT', o.get('gt_answer'),
                  'OK' if o.get('correct') else 'WRONG/NA', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out.append(dict(item=item['item_id'], template=tpl, clip=c['name'], error=repr(e)[:300]))
    return out


MATCH_PX = 6.0  # a pseudo point counts as matched to a GT track within this distance at t0
NB_PX, NB_K = 40.0, 6  # GT motion of a pseudo pixel: from up to NB_K GT tracks within NB_PX on the same GT instance


def gt_point_track(g, scene, f0, uv0, frames):
    """A GT track for an arbitrary pixel: its GT 3D point at source frame f0 (GT depth + camera), moved with the
    distance-weighted mean world motion of nearby GT tracks visible at f0 on the same GT instance (PointOdyssey GT
    tracks are dense surface samples, so this is GT scene flow interpolation). Visibility on each frame: the GT depth
    map at the projected pixel agrees with the point's depth within 5%. Returns uv (F, 2), r (F,), vis (F,),
    onscreen (F,), info."""
    a = gt_anno(scene)
    E, Ks = a['E'], a['K']
    u, v = int(round(uv0[0])), int(round(uv0[1]))
    z0 = float(g['depth'](f0)[v, u])
    if not 0.05 < z0 < 900:  # no GT surface at this pixel (sky / invalid depth)
        return None
    m0 = g['mask'](f0)
    Pc = z0 * np.linalg.inv(Ks[f0]) @ np.array([uv0[0], uv0[1], 1.0])
    X0 = np.linalg.inv(E[f0]) @ np.r_[Pc, 1]
    X0 = X0[:3]
    cand = np.nonzero(g['vis'][f0] & g['onscreen'][f0])[0]
    dd = np.linalg.norm(g['uv'][f0, cand] - uv0, axis=1)
    uu = np.clip(np.round(g['uv'][f0, cand]).astype(int), [0, 0], [g['W'] - 1, g['H'] - 1])
    same = m0[uu[:, 1], uu[:, 0]] == m0[v, u]
    ok = (dd <= NB_PX) & same
    if ok.any():
        o = np.argsort(np.where(ok, dd, np.inf))[:NB_K]
        o = o[ok[o]]
        w = 1 / (dd[o] + 1)
        w /= w.sum()
        J = cand[o]
        disp = (g['X'][frames][:, J] - g['X'][f0][J]) * w[None, :, None]
        X = X0 + disp.sum(1)
        nn = float(dd[o].min())
    elif (m0 == m0[v, u]).mean() > 0.10:  # scene background (floor, walls): static, the GT 3D point stays put
        J, nn = [], float('nan')
        X = np.repeat(X0[None], len(frames), 0)
    else:
        return None
    P = np.einsum('fij,fj->fi', E[frames, :3, :3], X) + E[frames, :3, 3]
    q = np.einsum('fij,fj->fi', Ks[frames], P)
    uv = q[:, :2] / np.maximum(q[:, 2:], 1e-6)
    on = (P[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < g['W']) & (uv[:, 1] >= 0) & (uv[:, 1] < g['H'])
    vis = np.zeros(len(frames), bool)
    for k, f in enumerate(frames):
        if on[k]:
            zz = g['depth'](int(f))[int(uv[k, 1]), int(uv[k, 0])]
            vis[k] = abs(zz - P[k, 2]) <= 0.05 * P[k, 2]
    info = dict(n_neighbours=int(len(J)), nn_px=nn, gt_z0=z0, static_background=len(J) == 0)
    return uv, np.linalg.norm(P, axis=1), vis, on, info


_anno = {}


def gt_anno(scene):
    if scene not in _anno:
        _anno.clear()
        a = np.load(f'{GT_PO}/{scene}/anno.npz')
        _anno[scene] = dict(E=a['extrinsics'].astype(np.float64), K=a['intrinsics'].astype(np.float64))
    return _anno[scene]


def eval_B_po(c, item):
    """GT re-labelling of a generated item: every pseudo point gets a GT track for its exact t0 pixel
    (gt_point_track), the template's label rule is applied to those GT tracks on the item's baked frames.
    Also reported: pseudo-vs-GT track pixel error, visibility agreement, and the nearest dataset GT track."""
    tpl = item['template']
    st = c['start']
    g = gt_po(c['scene'])
    pa = np.load(f"{PO_ROOT}/{item['video_id']}/anno.npz")
    letters = list(item['points'])
    fb = np.array(item['baked_frames'])
    t0, tn = item['t0'], item['tn']
    frames = np.unique(np.r_[fb, t0, tn]) + st
    U, R, V, O, pts = [], [], [], [], {}
    for L in letters:
        jp = item['track_indices'][L]
        uv0 = pa['trajs_2d'][t0, jp]
        tr = gt_point_track(g, c['scene'], t0 + st, uv0, frames)
        cand = np.nonzero(g['vis'][t0 + st] & g['onscreen'][t0 + st])[0]
        dn = float(np.linalg.norm(g['uv'][t0 + st, cand] - uv0, axis=1).min())
        if tr is None:
            pts[L] = dict(verifiable=False, nearest_gt_track_px=dn)
            continue
        uvg, rg, vg, og, info = tr
        sel = np.searchsorted(frames, fb + st)
        e = np.linalg.norm(pa['trajs_2d'][fb, jp] - uvg[sel], axis=1)
        ev = e[vg[sel]] if vg[sel].any() else e
        pts[L] = dict(verifiable=True, nearest_gt_track_px=dn, px_med=float(np.median(ev)), px_p90=float(np.percentile(ev, 90)),
                      vis_agree=float((pa['visibs'][fb, jp] == vg[sel]).mean()), **info)
        U.append(uvg), R.append(rg), V.append(vg), O.append(og)
    ps_ans = item['ranking_tn'] if tpl == 'T4' else item['answer']
    if len(U) < len(letters):
        return dict(gt_answer=None, correct=None, points=pts, note='a point has no GT neighbours on its instance')
    T = frames.max() + 1  # index arrays by source frame
    full = lambda arrs, shape, fill: np.stack([np.full((T,) + shape, fill, dtype=np.asarray(x).dtype) for x in arrs], 1)
    uv, r, vis, on = full(U, (2,), np.nan), full(R, (), np.nan), full(V, (), False), full(O, (), False)
    for k in range(len(letters)):
        uv[frames, k], r[frames, k], vis[frames, k], on[frames, k] = U[k], R[k], V[k], O[k]
    gt_ans, det = po_label(tpl, uv, vis, r, on, fb + st, t0 + st, tn + st, g['thr'], letters)
    if tpl == 'T3' and det.get('kind') == 'easy negative' and det['min_dist_px'] < ns('T3')['EASY_MIN'] * g['thr']:
        gt_ans = 'ambiguous'  # a near miss: GT would make neither a positive nor an easy negative of it
    ok = None if gt_ans == 'ambiguous' else answers_equal(tpl, ps_ans, gt_ans)  # GT itself gives no clear answer
    return dict(gt_answer=gt_ans, correct=ok, gt_detail=det, points=pts,
                points_matched=all(p.get('px_med', 99) <= 8 for p in pts.values()))


def eval_B_kitti(c, item):
    n6 = ns('T6')
    st = c['start']
    gd = gt_kitti(c['drive'])
    t0, te = item['t0'] + st, item['te'] + st
    ev = n6['events'](gd)
    same = [e for e in ev if e[0] == item['event']]
    near = min(same, key=lambda e: abs(e[1] - te)) if same else None
    res = dict(event=item['event'], gt_events=[list(e) for e in ev],
               gt_tracklets=os.path.exists(f"{GT_KR}/{c['drive'][:10]}/{c['drive']}_sync/tracklet_labels.xml"),
               event_matched=bool(near is not None and abs(near[1] - te) <= round(gd['fps'])),
               event_offset_frames=None if near is None else int(near[1] - te))
    have = set(gd['velo_frames'])
    if t0 not in have or te not in have:
        res.update(gt_answer=None, correct=None, note='GT LiDAR not packed for t0/te')
        return res
    L0 = n6['lidar'](gd, t0)
    letters = ['A', 'B', 'C']
    uvs = np.array([item['points'][L]['uv_t0'] for L in letters])
    dist, idx = L0['tree'].query(uvs, k=1)
    Xv = L0['Xv'][idx]
    on_obj = n6['in_boxes'](gd, t0, Xv)
    r_t0 = np.linalg.norm(L0['Xc'][idx], axis=1)
    rp = np.array([item['points'][L]['r'][0] for L in letters])
    r_te = np.linalg.norm(n6['xform'](n6['velo_t0_to_cam'](gd, t0, te), Xv), axis=1)
    rs = np.sort(r_te)
    gt_ans = letters[int(np.argmin(r_te))] if (rs[1] - rs[0]) / rs[0] >= 0.15 else 'ambiguous'  # T6 margin M = 0.15
    res.update(gt_answer=gt_ans, correct=None if gt_ans == 'ambiguous' else gt_ans == item['answer'],
               points={L: dict(lidar_dist_px=float(dist[k]), on_gt_object=bool(on_obj[k]), r_t0_gt=float(r_t0[k]),
                               r_t0_pseudo=float(rp[k]), r_te_gt=float(r_te[k])) for k, L in enumerate(letters)},
               points_matched=bool((dist <= 3).all() and not on_obj.any()))
    if near is not None and near[1] != te and near[1] in have:  # also: GT answer at the GT event frame
        r_ge = np.linalg.norm(n6['xform'](n6['velo_t0_to_cam'](gd, t0, near[1]), Xv), axis=1)
        res['gt_answer_at_gt_event'] = letters[int(np.argmin(r_ge))]
    return res


# ---------------------------------------------------------------- C: every selection candidate checked with GT
def cand_item(tpl, c, clip):
    """A selection candidate in the shape of a rendered item (baked frames, letters, pseudo answer)."""
    from common import bake_indices
    js = c['j'] if isinstance(c['j'], list) else [c['j']]
    letters = ['A', 'B', 'C'][:len(js)]
    if tpl == 'T6':
        fb = bake_indices(c['t0'], c['tn'], [c['t0'], c['te'], c['tn']])
        return dict(template=tpl, item_id=f"cand_{clip['name']}_{c['t0']}_{'_'.join(map(str, js))}", video_id=c['drive'],
                    baked_frames=fb, t0=c['t0'], te=c['te'], tn=c['tn'], event=c['event'], js=js, answer=None)
    fb = bake_indices(c['t0'], c['tn'], [c['t0'], c['tn']])
    pa = np.load(f"{PO_ROOT}/pseudo/{clip['name']}/anno.npz")
    E = pa['extrinsics'].astype(np.float64)
    P = np.einsum('tij,tnj->tni', E[:, :3, :3], pa['trajs_3d'][:, js]) + E[:, None, :3, 3]
    r = np.linalg.norm(P, axis=-1)
    if tpl == 'T4':
        ans = sorted(letters, key=lambda L: r[c['tn'], letters.index(L)])
    else:
        ans = c.get('label') or ('Farther' if c['rel'] > 0 else 'Closer')
    item = dict(template=tpl, item_id=f"cand_{clip['name']}_{c['t0']}_{'_'.join(map(str, js))}",
                video_id=f"pseudo/{clip['name']}", baked_frames=fb, t0=c['t0'], tn=c['tn'],
                track_indices=dict(zip(letters, js)), points={L: {} for L in letters}, answer=ans, kind=c.get('kind'))
    if tpl == 'T4':
        item['ranking_tn'] = ans
    return item


def eval_C(clips, tpls=('T1', 'T3', 'T4', 'T5')):
    """GT labels for all candidates in <ITEMS>/_work/<T>/cands/ (PO templates)."""
    by = {f"pseudo_{c['name']}": c for c in clips if c['src'] == 'po'}
    out = []
    for tpl in tpls:
        for f in sorted(glob.glob(f'{ITEMS}/_work/{tpl}/cands/*.json')):
            clip = by.get(os.path.basename(f)[:-5])
            if clip is None:
                continue
            for c in json.load(open(f)):
                item = cand_item(tpl, c, clip)
                try:
                    res = eval_B_po(clip, item)
                except Exception as e:
                    res = dict(error=repr(e)[:200])
                out.append(dict(template=tpl, clip=clip['name'], kind=c.get('kind') or c.get('label'),
                                pseudo_answer=item['answer'], **res))
        rs = [r for r in out if r['template'] == tpl and r.get('correct') is not None]
        print('C', tpl, f"{sum(r['correct'] for r in rs)}/{len(rs)} GT-decidable candidates agree", flush=True)
    return out


# ---------------------------------------------------------------- summary
def summarize(A, B):
    L = ['# Pseudo-GT vs GT', '',
         '## A: GT items re-answered from pseudo-GT', '', '| template | items | pseudo answer = GT answer |', '|---|---|---|']
    for tpl in ['T1', 'T3', 'T4', 'T5', 'T6']:
        rs = [r for r in A if r['template'] == tpl and 'error' not in r]
        L.append(f"| {tpl} | {len(rs)} | {sum(bool(r['correct']) for r in rs)}/{len(rs)} |")
    L += ['', '## B: pseudo-pipeline items re-labelled with GT', '',
          '| template | items | GT-verifiable | GT ambiguous (below margin) | pseudo answer = GT answer |', '|---|---|---|---|---|']
    for tpl in ['T1', 'T3', 'T4', 'T5', 'T6']:
        rs = [r for r in B if r['template'] == tpl and 'error' not in r]
        ver = [r for r in rs if r.get('gt_answer') is not None]
        amb = [r for r in ver if r['gt_answer'] == 'ambiguous']
        dec = [r for r in ver if r['gt_answer'] != 'ambiguous']
        L.append(f"| {tpl} | {len(rs)} | {len(ver)} | {len(amb)} | {sum(bool(r['correct']) for r in dec)}/{len(dec)} |")
    return '\n'.join(L) + '\n'


def summarize_B(B, tpls):
    L = ['## B: pseudo-pipeline items re-labelled with GT', '',
         '| template | items | GT-verifiable | GT ambiguous (below margin) | pseudo answer = GT answer |', '|---|---|---|---|---|']
    for tpl in tpls:
        rs = [r for r in B if r['template'] == tpl and 'error' not in r]
        ver = [r for r in rs if r.get('gt_answer') is not None]
        amb = [r for r in ver if r['gt_answer'] == 'ambiguous']
        dec = [r for r in ver if r['gt_answer'] != 'ambiguous']
        L.append(f"| {tpl} | {len(rs)} | {len(ver)} | {len(amb)} | {sum(bool(r['correct']) for r in dec)}/{len(dec)} |")
    n_err = sum('error' in r for r in B)
    return '\n'.join(L + ([f'', f'{n_err} items failed to evaluate (see the json)'] if n_err else [])) + '\n'


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    clips = json.load(open(CLIPS))
    os.makedirs(EVAL, exist_ok=True)
    if what == 'B' and len(sys.argv) > 3:  # eval_gt.py B T1,T3,T4 <tag>: incremental, B_<tag>.json + summary_<tag>.md
        tpls, tag = sys.argv[2].split(','), sys.argv[3]
        path = f'{EVAL}/B_{tag}.json'
        prev = json.load(open(path)) if os.path.exists(path) else []
        B = eval_B(clips, tpls, prev)
        json.dump(B, open(path + '.tmp', 'w'), indent=1)
        os.replace(path + '.tmp', path)
        s = summarize_B(B, tpls)
        open(f'{EVAL}/summary_{tag}.md', 'w').write(s)
        print(s)
        sys.exit(0)
    A = eval_A(clips) if what in ('A', 'both') else (json.load(open(f'{EVAL}/A.json')) if os.path.exists(f'{EVAL}/A.json') else [])
    json.dump(A, open(f'{EVAL}/A.json', 'w'), indent=1)
    B = eval_B(clips) if what in ('B', 'both') else (json.load(open(f'{EVAL}/B.json')) if os.path.exists(f'{EVAL}/B.json') else [])
    json.dump(B, open(f'{EVAL}/B.json', 'w'), indent=1)
    if what in ('C', 'both'):
        C = eval_C(clips)
        json.dump(C, open(f'{EVAL}/C.json', 'w'), indent=1)
    s = summarize(A, B)
    open(f'{EVAL}/summary.md', 'w').write(s)
    print(s)
