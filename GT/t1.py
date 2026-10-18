# T1 -- distance change (Dynamic Replica).
# Q: "Point A is marked at the start. After {n} seconds, is it closer to or farther from the camera
# than at the start?"  Balanced 5 Closer / 5 Farther, at most one item per video.
#   python t1.py --n_items 10 [--videos a,b,...] [--out_tag T1]
import argparse, json, os, shutil
import numpy as np
from common import OUT_ROOT, bake_indices, dr_sequences, euclid, in_bounds, load_dr, patch_rel_std
from render import N_MARKED, baked_size, render_item, ring_radius

M = 0.10  # relative margin, synthetic data
STD_MAX = 0.02  # relative depth std in 5x5 patch
DEPTH_AGREE = 0.02  # |depth map - track z| / z at t0 (guards the dataset visibility flag)
Q = 'Point A is marked at the start. After {n} seconds, is it closer to or farther from the camera than at the start?'

ap = argparse.ArgumentParser()
ap.add_argument('--n_items', type=int, default=10)
ap.add_argument('--videos', default='')
ap.add_argument('--out_tag', default='T1')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
out_root = f'{OUT_ROOT}/{args.out_tag}'
os.makedirs(out_root, exist_ok=True)

videos = args.videos.split(',') if args.videos else dr_sequences()
rng.shuffle(videos)
target = {'Closer': args.n_items // 2, 'Farther': args.n_items - args.n_items // 2}
done = {'Closer': 0, 'Farther': 0}
st = dict(videos_scanned=0, windows_scanned=0, windows_passed=0, points_tried=0, points_passed=0,
          reject=dict(not_visible=0, border=0, patch_std=0, depth_disagree=0, margin=0))
items = []

for vid in videos:
    if sum(done.values()) >= args.n_items:
        break
    need = [l for l in ('Closer', 'Farther') if done[l] < target[l]]
    d = load_dr(vid)
    st['videos_scanned'] += 1
    T, fps, W, H = d['n_frames'], d['fps'], d['W'], d['H']
    Wb, Hb, s = baked_size(H, W)
    pad = 3 * ring_radius(Wb) / s  # keep marker + label inside the frame (source px)
    r_all = euclid(d['P_cam'])
    found = None
    for n in rng.permutation([1, 2, 3]):
        L = int(round(n * fps))
        for t0 in rng.permutation(np.arange(0, T - L, 10)):
            tn = t0 + L
            st['windows_scanned'] += 1
            frames = bake_indices(t0, tn, [t0, tn])
            first = frames[:N_MARKED]
            ok = d['vis'][first].all(0) & d['vis'][tn]
            rel = (r_all[tn] - r_all[t0]) / r_all[t0]
            if not (ok & (np.abs(rel) >= M)).any():  # motion predicate: some tracked point changes >= m
                continue
            st['windows_passed'] += 1
            cand = rng.permutation(np.nonzero(ok)[0])[:400]
            dep0 = None
            for j in cand:
                lab = 'Closer' if rel[j] < 0 else 'Farther'
                if lab not in need:
                    continue
                st['points_tried'] += 1
                if not in_bounds(d['uv'][first, j], W, H, pad).all():
                    st['reject']['border'] += 1
                    continue
                if abs(rel[j]) < M:
                    st['reject']['margin'] += 1
                    continue
                if dep0 is None:
                    dep0 = d['depth'](t0)
                u, v = d['uv'][t0, j]
                if patch_rel_std(dep0, u, v) >= STD_MAX:
                    st['reject']['patch_std'] += 1
                    continue
                z = d['P_cam'][t0, j, 2]
                if abs(dep0[int(round(v)), int(round(u))] - z) / z > DEPTH_AGREE:
                    st['reject']['depth_disagree'] += 1
                    continue
                st['points_passed'] += 1
                found = (n, t0, tn, frames, j, lab)
                break
            if found:
                break
        if found:
            break
    if not found:
        continue
    n, t0, tn, frames, j, lab = found
    k = sum(done.values())
    item_id = f'{args.out_tag}_{k:02d}'
    out = f'{out_root}/{item_id}'
    if os.path.exists(out):
        shutil.rmtree(out)
    fr = np.array(frames)
    pts = {'A': dict(uv=d['uv'][fr, j], r=r_all[fr, j], vis=d['vis'][fr, j])}
    margin = abs(r_all[tn, j] - r_all[t0, j]) / r_all[t0, j]
    render_item(out, d, frames, pts, [t0, tn], Q.format(n=n), ['Closer', 'Farther'], lab,
                dict(template='T1', item_id=item_id, n_seconds=int(n), t0=int(t0), tn=int(tn), track_index=int(j),
                     instance=int(d['inst'][j]), r_t0=float(r_all[t0, j]), r_tn=float(r_all[tn, j]),
                     margin_achieved=float(margin),
                     thresholds=dict(m=M, patch_std=STD_MAX, depth_agree=DEPTH_AGREE, marker_pad_px=float(pad))))
    done[lab] += 1
    items.append(dict(item_id=item_id, video=vid, label=lab, n=int(n), margin=float(margin)))
    print(item_id, vid, lab, f'n={n} t0={t0} r {r_all[t0, j]:.2f}->{r_all[tn, j]:.2f} m={margin:.3f}', flush=True)

json.dump(dict(stats=st, labels=done, items=items), open(f'{out_root}/stats.json', 'w'), indent=1)
print(json.dumps(st), done)
