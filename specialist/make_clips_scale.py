# Scale-up clips (target: 100 items per template): append generic windows to <ROOT>/clips.json and write the
# prioritised work lists of the two overnight jobs.
#   PO (T1/T3/T4/T5): every packed PointOdyssey scene is tiled into non-overlapping L_PO-frame windows that don't
#     overlap an existing clip. Every PO clip serves all four PO templates (assign takes one item per clip per
#     template). Windows are ranked by how many production GT candidates lie inside them (job A: T1/T3/T4 candidates,
#     job B: T5), and only the best N_PO are kept (disk: ~0.35 GB per clip).
#   KITTI (T6): 100-frame windows centred on GT OXTS events (stop / turn, the walkthrough's rules) on all drives,
#     non-overlapping with each other and with existing clips. Images and LiDAR (eval B) of windows that aren't packed
#     are fetched by fetch_kitti_more.py inside job B. Drives without tracklets: eval B marks gt_tracklets=False.
#   python make_clips_scale.py [--n_po 200] [--dry]
# Writes <ROOT>/clips.json (old clips unchanged, new ones appended), <ROOT>/scale/work_A.txt, work_B.txt, fetch_B.txt.
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from build import CLIPS, ROOT
from blk import PO

PROD = '/shared/rsaas/joeyw3/pvideo'
KR = '/shared/rsaas/joeyw3/data/kitti_raw'
SCALE = f'{ROOT}/scale'
L_PO = 192  # 8 s at 24 fps: fits the longest PO window (T5, 6 s) with 1 s on each side
L_KITTI, PRE_KITTI = 100, 50  # 10 s at 10 fps, event at frame 50 (the old T6 clips are 100 frames too)
MIN_GAP = 0  # windows may touch but not overlap
STOP_V, STOP_FROM, STOP_LOOK = 0.5, 3.0, 5.0  # = walkthrough_kitti_event.py
TURN_W, TURN_HOLD = 0.1, 0.5


def overlaps(a, b, slack=MIN_GAP):
    return a[0] <= b[1] + slack and b[0] <= a[1] + slack


def po_scenes():
    return sorted(os.path.relpath(os.path.dirname(f), PO) for f in glob.glob(f'{PO}/*/*/.complete'))


def po_windows(clips, n_po):
    taken = {}
    for c in clips:
        if c['src'] == 'po':
            taken.setdefault(c['scene'], []).append((c['start'], c['end']))
    cands = {t: [k for f in glob.glob(f'{PROD}/_work/{t}/cands/*.json') for k in json.load(open(f))]
             for t in ['T1', 'T3', 'T4', 'T5']}
    by_scene = {}
    for t, ks in cands.items():
        for k in ks:
            by_scene.setdefault(k['scene'], []).append((t, k['t0'], k['tn']))
    wins = []
    for sc in po_scenes():
        T = len(np.load(f'{PO}/{sc}/anno.npz')['intrinsics'])
        for s in range(0, T - L_PO + 1, L_PO):
            w = (s, s + L_PO - 1)
            if any(overlaps(w, x) for x in taken.get(sc, [])):
                continue
            inside = [t for t, a, b in by_scene.get(sc, []) if w[0] <= a and b <= w[1]]
            wins.append(dict(scene=sc, start=w[0], end=w[1], score_A=sum(t in ('T1', 'T3', 'T4') for t in inside),
                             score_B=sum(t == 'T5' for t in inside)))
    print(len(wins), 'free PO windows in', len(po_scenes()), 'scenes;', sum(w['score_A'] + w['score_B'] > 0 for w in wins),
          'contain GT candidates', flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(wins))  # random tie-break, then best first
    wins = [wins[i] for i in order]
    wins.sort(key=lambda w: -(w['score_A'] + w['score_B']))
    return wins[:n_po]


def kitti_events(drive):
    """GT OXTS events [(kind, frame)] with the walkthrough rules (fps 10)."""
    d = f'{KR}/{drive[:10]}/{drive}_sync/oxts/data'
    O = np.array([np.loadtxt(f'{d}/{f}') for f in sorted(os.listdir(d))])
    v, w, fps = np.hypot(O[:, 6], O[:, 7]), O[:, 22], 10
    T, hold, look, sec = len(v), int(round(TURN_HOLD * fps)), int(round(STOP_LOOK * fps)), fps
    ev = [('stop', t) for t in range(1, T) if v[t] < STOP_V <= v[t - 1] and (v[max(0, t - look):t] > STOP_FROM).any()]
    for t in range(1, T - hold):
        seg = w[t:t + hold]
        if abs(w[t]) >= TURN_W > abs(w[t - 1]) and (np.abs(seg) >= TURN_W).all() and (np.sign(seg) == np.sign(w[t])).all():
            ev.append(('turn_left' if w[t] > 0 else 'turn_right', t))
    return T, sorted([e for e in ev if sec <= e[1] <= T - 1 - sec], key=lambda e: e[1])


def kitti_windows(clips):
    cache = f'{SCALE}/kitti_events.json'
    # drives without tracklet_labels.xml are used too: eval B then can't flag GT points on objects (gt_tracklets=False)
    drives = sorted(os.path.basename(d)[:-5] for d in glob.glob(f'{KR}/*/*_sync') if os.path.isdir(f'{d}/oxts/data'))
    ev = json.load(open(cache)) if os.path.exists(cache) else {}
    for drv in drives:
        if drv not in ev:
            T, e = kitti_events(drv)
            ev[drv] = dict(n=T, events=e)
            print('events', drv, T, 'frames', len(e), 'events', flush=True)
    json.dump(ev, open(cache, 'w'))
    taken = {}
    for c in clips:
        if c['src'] == 'kitti':
            taken.setdefault(c['drive'], []).append((c['start'], c['end']))
    wins = []
    for drv in drives:
        T = ev[drv]['n']
        for kind, t in ev[drv]['events']:
            s = min(max(0, t - PRE_KITTI), max(0, T - L_KITTI))
            w = (s, min(T, s + L_KITTI) - 1)
            if any(overlaps(w, x) for x in taken.get(drv, [])):
                continue
            taken.setdefault(drv, []).append(w)
            wins.append(dict(drive=drv, start=w[0], end=w[1], event=kind, te=t))
    print(len(wins), 'free KITTI event windows on', len({w['drive'] for w in wins}), 'drives', flush=True)
    return wins


def has_images(drv):
    return os.path.exists(f'{KR}/{drv[:10]}/{drv}_sync/image_02.idx.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_po', type=int, default=200)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    os.makedirs(SCALE, exist_ok=True)
    clips = json.load(open(CLIPS))
    assert not any(c['name'].startswith(('P', 'K')) for c in clips), 'scale clips already in clips.json'
    po = po_windows(clips, a.n_po)
    new = [dict(name=f'P{i + 1:04d}', template='PO', src='po', scene=w['scene'], start=w['start'], end=w['end'],
                gt_item=None, gt_answer=None, score_A=w['score_A'], score_B=w['score_B'], probes=[])
           for i, w in enumerate(po)]
    kw = kitti_windows(clips)
    for i, w in enumerate(kw):
        name = f'K{i + 1:05d}'  # 6 characters: the KITTI date dir is drive_id[:10] = 'pgt' + name + '_'
        new.append(dict(name=name, template='T6', src='kitti', drive=w['drive'], start=w['start'], end=w['end'],
                        drive_id=f"pgt{name}_{w['drive']}", gt_item=None, gt_answer=None, t0=0, tn=w['end'] - w['start'],
                        te=w['te'] - w['start'], event=w['event'], probes=[]))
        assert new[-1]['drive_id'][:10] == f'pgt{name}_'
    pn = [c for c in new if c['src'] == 'po']
    kn = [c for c in new if c['src'] == 'kitti']
    work_A = [c['name'] for c in sorted(pn, key=lambda c: -c['score_A'])]  # stable: ties keep the random order
    work_B = [c['name'] for c in kn] + [c['name'] for c in sorted(pn, key=lambda c: -c['score_B'])]
    fetch = {}
    for c in kn:  # frames needed: images and LiDAR (eval B) of the clip window
        fetch.setdefault(c['drive'], []).append(f"{c['start']}-{c['end']}")
    print(f"PO: {len(pn)} new clips ({sum(c['score_A'] > 0 for c in pn)} with T1/T3/T4 GT candidates, "
          f"{sum(c['score_B'] > 0 for c in pn)} with T5); KITTI: {len(kn)} new clips on {len(fetch)} drives "
          f"({sum(not has_images(d) for d in fetch)} drives without images, "
          f"{sum(not os.path.exists(f'{KR}/{d[:10]}/{d}_sync/tracklet_labels.xml') for d in fetch)} without tracklets)")
    if a.dry:
        sys.exit(0)
    json.dump(clips + new, open(CLIPS, 'w'), indent=1)
    open(f'{SCALE}/work_A.txt', 'w').write('\n'.join(work_A) + '\n')
    open(f'{SCALE}/work_B.txt', 'w').write('\n'.join(work_B) + '\n')
    open(f'{SCALE}/fetch_B.txt', 'w').write(''.join(f"{d}:{','.join(r)}\n" for d, r in fetch.items()))
    print(len(clips), '+', len(new), 'clips ->', CLIPS)
