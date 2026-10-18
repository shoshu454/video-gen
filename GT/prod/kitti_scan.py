# Find KITTI drives on disk (OXTS + tracklets, not yet packed) that have T6 events; print fetch args
# "<drive>:<lo>-<hi>,..." covering t_e -45..+45 frames (as fetch_kitti_blk.py expects).
import os, sys, numpy as np
KR = '/shared/rsaas/joeyw3/data/kitti_raw'
STOP_V, STOP_FROM, STOP_LOOK, TURN_W, TURN_HOLD, fps = 0.5, 3.0, 5.0, 0.1, 0.5, 10.0
def events(O):
    v, w = np.hypot(O[:, 6], O[:, 7]), O[:, 22]
    T, hold, look, sec = len(v), int(round(TURN_HOLD * fps)), int(round(STOP_LOOK * fps)), int(round(fps))
    ev = []
    for t in range(1, T):
        if v[t] < STOP_V <= v[t - 1] and (v[max(0, t - look):t] > STOP_FROM).any():
            ev.append(('stop', t))
    for t in range(1, T - hold):
        seg = w[t:t + hold]
        if abs(w[t]) >= TURN_W > abs(w[t - 1]) and (np.abs(seg) >= TURN_W).all() and (np.sign(seg) == np.sign(w[t])).all():
            ev.append(('turn', t))
    return [e for e in ev if sec <= e[1] <= T - 1 - sec], T
for date in sorted(os.listdir(KR)):
    if not date.startswith('2011'): continue
    for dd in sorted(os.listdir(f'{KR}/{date}')):
        p = f'{KR}/{date}/{dd}'
        if not dd.endswith('_sync') or not os.path.isdir(f'{p}/oxts/data'): continue
        packed = os.path.exists(f'{p}/.blk_complete'); trk = os.path.exists(f'{p}/tracklet_labels.xml')
        O = np.array([np.loadtxt(f'{p}/oxts/data/{f}') for f in sorted(os.listdir(f'{p}/oxts/data'))])
        ev, T = events(O)
        rng = []
        for _, t in ev:
            lo, hi = max(0, t - 45), min(T - 1, t + 45)
            if rng and lo <= rng[-1][1] + 1: rng[-1][1] = max(rng[-1][1], hi)
            else: rng.append([lo, hi])
        print('DRIVE', dd[:-5], 'packed' if packed else 'unpacked', 'trk' if trk else 'notrk', len(ev), T,
              ','.join(f'{a}-{b}' for a, b in rng), flush=True)
