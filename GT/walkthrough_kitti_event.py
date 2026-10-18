# Stage-by-stage walkthrough clips for T6 (event-anchored closest) on KITTI raw.
# Same layout as walkthrough_pointodyssey.py (T1): one drive, one window, points A, B, C across stages 0-7.
# Every number comes from KITTI GT: Velodyne returns, OXTS ego pose / speed / yaw rate, calibration, tracklets.
#   Q: "When {event}, which of A, B and C is closest to the camera?"  A / B / C.
#   events (OXTS): "the car comes to a stop" (speed < 0.5 m/s after > 3 m/s in the previous 5 s),
#                  "the car starts turning left/right" (|yaw rate| crosses 0.1 rad/s and holds >= 0.5 s).
#   points: static background LiDAR returns at t0 outside all tracklet boxes, propagated with the OXTS ego pose;
#           visibility = z-buffer against each frame's own LiDAR (5% relative); round trip t0 -> t_e -> t0 <= 3 px.
# Data: fetch_kitti_blk.py packs image_02 (all frames) and velodyne (event +-45 frames) per drive.
# Writes to <OUT_ROOT>/walkthrough_T6/item<k>/: stage{k}_<name>.mp4, stage{k}.png, walkthrough_full.mp4, meta.json
#   python walkthrough_kitti_event.py --select --drives 2011_09_26_drive_0011   # write sel/<drive>.json
#   python walkthrough_kitti_event.py --merge                                   # selection.json (5 picks)
#   python walkthrough_kitti_event.py --pick k                                  # render pick k (0-4)
import argparse, json, os, sys, textwrap
import xml.etree.ElementTree as ET
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blk import names, read
from common import OUT_ROOT, bake_indices, in_bounds
from render import COLORS, N_MARKED, baked_size, draw_marker, ring_radius, write_mp4
import io

ap = argparse.ArgumentParser()
ap.add_argument('--select', action='store_true')
ap.add_argument('--merge', action='store_true')
ap.add_argument('--drives', default='', help='comma list; default every drive with packed data')
ap.add_argument('--pick', type=int, default=0)
args = ap.parse_args()
TPL = 'T6'
OUT = f'{OUT_ROOT}/walkthrough_{TPL}'
KR = '/shared/rsaas/joeyw3/data/kitti_raw'
os.makedirs(OUT, exist_ok=True)

FPS_OUT = 10
W_OUT, H_BAN, H_BODY = 960, 64, 540
LETTERS = ['A', 'B', 'C']
M = 0.15  # relative margin |r_i - r_j| / min(r_i, r_j), KITTI
M_SHOW = 0.20  # walkthrough item: larger margins at t_e, for visual clarity
N_SEC = 4  # window length; t_e has >= 1 s before and after
PRE = [1.0, 1.5, 2.0, 2.5, 3.0]  # s from t0 to t_e (tried per event)
STOP_V, STOP_FROM, STOP_LOOK = 0.5, 3.0, 5.0  # m/s, m/s, s
TURN_W, TURN_HOLD = 0.1, 0.5  # rad/s, s
ZB_TOL = 0.05  # z-buffer: relative depth tolerance
ZB_RAD = 4  # z-buffer: LiDAR returns within this many pixels of the projection
RT_MAX = 3.0  # px, round trip t0 -> t_e -> t0
STD_MAX = 0.02  # relative z std of the LiDAR returns in the 5x5 patch
BOX_PAD = 0.3  # m, tracklet boxes are grown by this before excluding points
R_MAX = 45.0  # m, points farther than this are not used (sparse, hard to see)
STEP_SEC = 0.5
N_TRIPLES = 200000
PLAN = dict(n=5, min_changed=3)  # spec: >= half of the items have a different closest point at t_e than at t0

GRAY = (150, 150, 150)
PASS_COL, FAIL_COL = (44, 160, 44), (214, 39, 40)
FONT_DIR = f'{matplotlib.get_data_path()}/fonts/ttf'
STAGES = [(0, 'standardize', 'Standardize', 'Common format: RGB, calibration, OXTS ego pose, LiDAR distance to camera.'),
          (1, 'geometry', 'Geometry', 'Static LiDAR points propagated with the ego pose; visibility by z-buffer.'),
          (2, 'clips', 'Clips', 'Detect events from OXTS; keep windows with ≥ 1 s around an event.'),
          (3, 'regions', 'Regions', 'Static background only: drop LiDAR returns inside tracklet boxes.'),
          (4, 'points', 'Points', 'Place the query points; keep them only if they pass the checks.'),
          (5, 'labels', 'Labels', 'Compute the answer from GT distance at the event frame.'),
          (6, 'render', 'Render', 'Bake frames; mark the points on the first 3 frames only.'),
          (7, 'question', 'Question', 'Final QA item.')]
Q = 'When {event}, which of A, B and C is closest to the camera?'
EVENT_TEXT = {'stop': 'the car comes to a stop', 'turn_left': 'the car starts turning left',
              'turn_right': 'the car starts turning right'}


# ---------------------------------------------------------------- data + geometry
def read_calib(path):
    out = {}
    for line in open(path):
        k, _, v = line.partition(':')
        try:
            out[k] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass
    return out


def rt(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R).reshape(3, 3), np.asarray(t).ravel()
    return T


def oxts_poses(O):
    """KITTI devkit convention: Mercator position with the scale of the first frame, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    er = 6378137.0
    scale = np.cos(O[0, 0] * np.pi / 180)
    out = []
    for o in O:
        tx = scale * o[1] * np.pi * er / 180
        ty = scale * er * np.log(np.tan((90 + o[0]) * np.pi / 360))
        cr, sr, cp, sp, cy, sy = np.cos(o[3]), np.sin(o[3]), np.cos(o[4]), np.sin(o[4]), np.cos(o[5]), np.sin(o[5])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        out.append(rt(Rz @ Ry @ Rx, [tx, ty, o[2]]))
    T = np.array(out)
    return np.linalg.inv(T[0]) @ T  # relative to the first frame


def tracklets(path):
    out = []
    for it in ET.parse(path).getroot().find('tracklets').findall('item'):
        poses = [[float(p.find(k).text) for k in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')]
                 for p in it.find('poses').findall('item')]
        out.append(dict(type=it.find('objectType').text, h=float(it.find('h').text), w=float(it.find('w').text),
                        l=float(it.find('l').text), first=int(it.find('first_frame').text), poses=np.array(poses)))
    return out


def load(drive):
    date = drive[:10]
    dd = f'{KR}/{date}/{drive}_sync'
    cc = read_calib(f'{KR}/{date}/calib_cam_to_cam.txt')
    vc = read_calib(f'{KR}/{date}/calib_velo_to_cam.txt')
    iv = read_calib(f'{KR}/{date}/calib_imu_to_velo.txt')
    P2 = cc['P_rect_02'].reshape(3, 4)
    K = P2[:, :3].copy()
    R0 = np.eye(4)
    R0[:3, :3] = cc['R_rect_00'].reshape(3, 3)
    T_cam2_velo = rt(np.eye(3), np.linalg.inv(K) @ P2[:, 3]) @ R0 @ rt(vc['R'], vc['T'])  # velo -> rectified cam 2
    T_velo_imu = rt(iv['R'], iv['T'])
    ox = sorted(os.listdir(f'{dd}/oxts/data'))
    O = np.array([np.loadtxt(f'{dd}/oxts/data/{f}') for f in ox])
    T_w_imu = oxts_poses(O)
    T_w_velo = T_w_imu @ np.linalg.inv(T_velo_imu)
    img_names = names(dd, 'image_02')
    vel_names = names(dd, 'velodyne')
    ts = [l.strip() for l in open(f'{dd}/image_02_timestamps.txt')]
    sec = np.array([int(t[11:13]) * 3600 + int(t[14:16]) * 60 + float(t[17:]) for t in ts])
    W, H = [int(x) for x in cc['S_rect_02']]

    def image(i):
        return np.array(Image.open(io.BytesIO(read(dd, 'image_02', img_names[i]))).convert('RGB'))

    def velo(i):
        return np.frombuffer(read(dd, 'velodyne', f'{i:010d}.bin'), np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

    d = dict(drive=drive, n_frames=len(img_names), fps=float(round(1 / np.median(np.diff(sec)), 3)), W=W, H=H, K=K,
             T_cam2_velo=T_cam2_velo, T_w_velo=T_w_velo, O=O, image=image, velo=velo,
             velo_frames=sorted(int(n[:10]) for n in vel_names),
             speed=np.hypot(O[:, 6], O[:, 7]), yaw_rate=O[:, 22], trk=tracklets(f'{dd}/tracklet_labels.xml'))
    d['lidar_cache'] = {}
    return d


def xform(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def velo_t0_to_cam(d, t0, t):
    """4x4 mapping velodyne coordinates at frame t0 to camera-2 coordinates at frame t (static world points)."""
    return d['T_cam2_velo'] @ np.linalg.inv(d['T_w_velo'][t]) @ d['T_w_velo'][t0]


def project(d, Xc):
    z = Xc[:, 2]
    uv = Xc @ d['K'].T
    uv = uv[:, :2] / np.maximum(z, 1e-6)[:, None]
    ok = (z > 0.5) & in_bounds(uv, d['W'], d['H'])
    return uv, ok


def lidar(d, t):
    """This frame's own returns in the image: velodyne xyz, camera xyz, pixel, and a KD-tree on the pixels."""
    c = d['lidar_cache']
    if t not in c:
        Xv = d['velo'](t)
        Xv = Xv[Xv[:, 0] > 0.5]
        Xc = xform(d['T_cam2_velo'], Xv)
        uv, ok = project(d, Xc)
        c[t] = dict(Xv=Xv[ok], Xc=Xc[ok], uv=uv[ok], tree=cKDTree(uv[ok]))
        if len(c) > 120:
            c.pop(next(iter(c)))
    return c[t]


def zbuffer(d, t, Xc):
    """Visible iff on screen, some return within ZB_RAD px has depth within ZB_TOL of the point (a surface is
    there) and no return within ZB_RAD px is more than ZB_TOL closer (nothing in front)."""
    uv, on = project(d, Xc)
    L = lidar(d, t)
    dist, idx = L['tree'].query(np.where(on[:, None], uv, -1e4), k=16, distance_upper_bound=ZB_RAD)
    has = idx < len(L['uv'])
    zl = np.where(has, L['Xc'][np.minimum(idx, len(L['uv']) - 1), 2], np.nan)
    z = Xc[:, 2:3]
    support = (has & (np.abs(zl - z) <= ZB_TOL * z)).any(1)
    front = (has & (zl < z * (1 - ZB_TOL))).any(1)
    return on & support & ~front, uv, on


def boxes_at(d, t):
    """Tracklet boxes at frame t: list of (type, pose (6,), (h, w, l))."""
    out = []
    for tr in d['trk']:
        k = t - tr['first']
        if 0 <= k < len(tr['poses']):
            out.append((tr['type'], tr['poses'][k], (tr['h'], tr['w'], tr['l'])))
    return out


def in_boxes(d, t, Xv):
    inside = np.zeros(len(Xv), bool)
    for _, p, (h, w, l) in boxes_at(d, t):
        c, s = np.cos(-p[5]), np.sin(-p[5])
        q = Xv - p[:3]
        x, y = c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]
        inside |= (np.abs(x) <= l / 2 + BOX_PAD) & (np.abs(y) <= w / 2 + BOX_PAD) & (q[:, 2] >= -BOX_PAD) & (q[:, 2] <= h + BOX_PAD)
    return inside


def box_corners(p, h, w, l):
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    c, s = np.cos(p[5]), np.sin(p[5])
    return np.stack([c * x - s * y + p[0], s * x + c * y + p[1], z + p[2]], 1)


def events(d):
    """[(kind, frame)] from OXTS with the rules above; t_e must have >= 1 s of video before and after."""
    fps, v, w = d['fps'], d['speed'], d['yaw_rate']
    T, hold, look, sec = len(v), int(round(TURN_HOLD * fps)), int(round(STOP_LOOK * fps)), int(round(fps))
    ev = []
    for t in range(1, T):
        if v[t] < STOP_V <= v[t - 1] and (v[max(0, t - look):t] > STOP_FROM).any():
            ev.append(('stop', t))
    for t in range(1, T - hold):
        seg = w[t:t + hold]
        if abs(w[t]) >= TURN_W > abs(w[t - 1]) and (np.abs(seg) >= TURN_W).all() and (np.sign(seg) == np.sign(w[t])).all():
            ev.append(('turn_left' if w[t] > 0 else 'turn_right', t))
    return sorted([e for e in ev if sec <= e[1] <= T - 1 - sec], key=lambda e: e[1])


def windows(d, ev):
    """Every 0.5 s: a window passes iff it contains an event with >= 1 s before and after, and LiDAR is on disk."""
    L = int(round(N_SEC * d['fps']))
    step = int(round(STEP_SEC * d['fps']))
    sec = int(round(d['fps']))
    have = set(d['velo_frames'])
    out = []
    for t0 in range(0, d['n_frames'] - L, step):
        tn = t0 + L
        e = [x for x in ev if t0 + sec <= x[1] <= tn - sec]
        lid = all(t in have for t in range(t0, tn + 1))
        out.append(dict(t0=t0, tn=tn, events=e, lidar=lid, passed=bool(e) and lid))
    return out


def rel_margin(ri, rj):
    return np.abs(ri - rj) / np.minimum(ri, rj)


def marker_pad(d):
    Wb, Hb, s = baked_size(d['H'], d['W'])
    return 3 * ring_radius(Wb) / s


def candidates_t0(d, t0, te, fb):
    """Static-background LiDAR returns at t0 and every check on them. Returns a dict of per-point arrays."""
    L0 = lidar(d, t0)
    Xv0, uv0 = L0['Xv'], L0['uv']
    n = len(Xv0)
    box = in_boxes(d, t0, Xv0)
    # tracks: camera coordinates of each static point on every baked frame (and t_e)
    Xc = {int(t): xform(velo_t0_to_cam(d, t0, int(t)), Xv0) for t in set(fb.tolist()) | {te}}
    vis = {t: zbuffer(d, t, Xc[t]) for t in Xc}
    r = {t: np.linalg.norm(Xc[t], axis=1) for t in Xc}
    # patch: relative z std of the returns within the 5x5 patch at t0 (the point itself included)
    dist, idx = L0['tree'].query(uv0, k=12, distance_upper_bound=2.0 * np.sqrt(2))
    has = idx < n
    zz = np.where(has, L0['Xc'][np.minimum(idx, n - 1), 2], np.nan)
    patch = np.nanstd(zz, 1) / np.nanmean(zz, 1)
    patch[has.sum(1) < 2] = np.inf
    # round trip t0 -> t_e -> t0: nearest t_e return to the propagated pixel (<= 3 px, depth within ZB_TOL),
    # moved back to t0 with the ego pose and projected
    Le = lidar(d, te)
    uve = vis[te][1]
    de, ie = Le['tree'].query(np.where(vis[te][2][:, None], uve, -1e4), k=1, distance_upper_bound=RT_MAX)
    okr = ie < len(Le['uv'])
    rt_err = np.full(n, np.inf)
    if okr.any():
        ze = Le['Xc'][ie[okr], 2]
        okz = np.abs(ze - Xc[te][okr, 2]) <= ZB_TOL * Xc[te][okr, 2]
        back = xform(velo_t0_to_cam(d, te, t0), Le['Xv'][ie[okr]])  # t_e velo -> t0 camera (inverse ego motion)
        uvb, _ = project(d, back)
        e = np.linalg.norm(uvb - uv0[okr], axis=1)
        rt_err[np.nonzero(okr)[0][okz]] = e[okz]
    return dict(Xv=Xv0, uv0=uv0, box=box, Xc=Xc, vis=vis, r=r, patch=patch, rt_err=rt_err)


# ---------------------------------------------------------------- selection
if args.select:
    drives = args.drives.split(',') if args.drives else sorted(
        os.path.basename(p)[:-5] for date in os.listdir(KR) if os.path.isdir(f'{KR}/{date}')
        for p in [f'{KR}/{date}/{x}' for x in os.listdir(f'{KR}/{date}')] if os.path.exists(f'{p}/.blk_complete'))
    os.makedirs(f'{OUT}/sel', exist_ok=True)
    for drv in drives:
        d = load(drv)
        W, H, fps = d['W'], d['H'], d['fps']
        ev = events(d)
        pad = marker_pad(d)
        L = int(round(N_SEC * fps))
        print(drv, d['n_frames'], 'fps', fps, 'events', ev, 'lidar', d['velo_frames'][0], d['velo_frames'][-1], flush=True)
        rng = np.random.default_rng(0)
        cands = []
        for kind, te in ev:
            for pre in PRE:
                t0 = te - int(round(pre * fps))
                tn = t0 + L
                if t0 < 0 or tn >= d['n_frames'] or tn - te < round(fps) or \
                        not all(t in set(d['velo_frames']) for t in range(t0, tn + 1)):
                    continue
                fb = np.array(bake_indices(t0, tn, [t0, te, tn]))
                c = candidates_t0(d, t0, te, fb)
                vis_all = np.stack([c['vis'][int(t)][0] for t in fb[:N_MARKED]] + [c['vis'][te][0]]).all(0)
                marked_in = np.stack([in_bounds(c['vis'][int(t)][1], W, H, pad) for t in fb[:N_MARKED]]).all(0)
                ok = ~c['box'] & vis_all & marked_in & (c['patch'] < STD_MAX) & (c['rt_err'] <= RT_MAX) \
                    & (c['r'][te] <= R_MAX) & (c['r'][t0] <= R_MAX) & (c['r'][te] >= 2.0)
                pool = np.nonzero(ok)[0]
                print(f'  {kind}@{te} t0={t0}: {len(c["uv0"])} returns, {int((~c["box"]).sum())} static, '
                      f'{len(pool)} pass point checks', flush=True)
                if len(pool) < 3:
                    continue
                J = rng.choice(pool, (N_TRIPLES, 3))
                J = J[(J[:, 0] != J[:, 1]) & (J[:, 0] != J[:, 2]) & (J[:, 1] != J[:, 2])]
                re, r0 = c['r'][te][J], c['r'][t0][J]
                pr = [(0, 1), (0, 2), (1, 2)]
                me = np.min([rel_margin(re[:, a], re[:, b]) for a, b in pr], 0)
                uvm = np.stack([c['vis'][int(t)][1][J] for t in fb[:N_MARKED]])  # (3, n, 3, 2)
                sp = np.min([np.linalg.norm(uvm[:, :, a] - uvm[:, :, b], axis=-1).min(0) for a, b in pr], 0)
                ce, c0 = re.argmin(1), r0.argmin(1)
                changed = ce != c0
                # t0 margin between the two closest points (no rule: with static points and forward motion a new point
                # can only become closest at t_e if the t0 distances were within ~13%, see README)
                r0s = np.sort(r0, 1)
                m0 = (r0s[:, 1] - r0s[:, 0]) / r0s[:, 0]
                ground = (c['Xv'][J][:, :, 2] < -1.3).sum(1)  # velodyne z; the road is at about -1.73 m
                ok3 = (me >= np.where(changed, M, M_SHOW)) & (sp >= pad)
                path = float(np.linalg.norm(d['T_w_velo'][te][:3, 3] - d['T_w_velo'][t0][:3, 3]))  # ego travel t0 -> t_e
                score = np.minimum(me, 0.6) + 0.6 * changed + 0.3 * np.minimum(m0, 0.1) - 0.15 * ground \
                    + 0.2 * np.minimum(sp / 150, 1) + 0.04 * min(path, 15)
                print(f'    triples: {len(J)} sampled, {int(ok3.sum())} pass, {int((ok3 & changed).sum())} with a changed closest',
                      flush=True)
                best = list(np.argsort(-np.where(ok3 & ~changed, score, -9))[:1]) + \
                    list(np.argsort(-np.where(ok3 & changed, score, -9))[:1])  # best of each type
                for q in best:
                    if ok3[q]:
                        cands.append(dict(drive=drv, event=kind, te=int(te), t0=int(t0), tn=int(tn),
                                          j=[int(x) for x in J[q]], changed=bool(changed[q]), min_margin_te=float(me[q]),
                                          margin_t0=float(m0[q]), n_ground=int(ground[q]), score=float(score[q]), ego_path_m=path,
                                          closest_t0=LETTERS[c0[q]], closest_te=LETTERS[ce[q]]))
                        print('  cand', json.dumps(cands[-1]), flush=True)
        json.dump(cands, open(f'{OUT}/sel/{drv}.json', 'w'), indent=1)
    sys.exit(0)

if args.merge:
    cands = [c for f in sorted(os.listdir(f'{OUT}/sel')) for c in json.load(open(f'{OUT}/sel/{f}'))]
    cands.sort(key=lambda c: -c['score'])
    best = {}  # best candidate per (drive, event kind)
    for c in cands:
        best.setdefault((c['drive'], c['event']), c)
    pool = sorted(best.values(), key=lambda c: -c['score'])
    picks, used = [], set()

    def take(pred, n):
        for c in pool:
            if len([p for p in picks if pred(p)]) >= n:
                return
            if c['drive'] not in used and pred(c) and c not in picks:
                picks.append(c)
                used.add(c['drive'])

    take(lambda c: c['event'] == 'stop', 2)  # variety: 2 stops, then turns left and right
    take(lambda c: c['event'] == 'turn_left', 1)
    take(lambda c: c['event'] == 'turn_right', 1)
    take(lambda c: True, PLAN['n'])
    # balance: >= min_changed items whose closest point changes between t0 and t_e (swap in the best such ones)
    for c in [c for c in pool if c['changed'] and c['drive'] not in used]:
        if sum(p['changed'] for p in picks) >= PLAN['min_changed']:
            break
        worst = min([p for p in picks if not p['changed']], key=lambda p: p['score'])
        picks[picks.index(worst)] = c
        used.discard(worst['drive'])
        used.add(c['drive'])
    json.dump(dict(picks=picks, candidates=cands), open(f'{OUT}/selection.json', 'w'), indent=1)
    print('picks:', [(p['drive'], p['event'], p['te'], 'changed' if p['changed'] else 'same') for p in picks], flush=True)
    sys.exit(0)


# ---------------------------------------------------------------- drawing helpers
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans{'-Bold' if bold else ''}.ttf", size)
    return _fonts[key]


def text(img, items):
    """items: [(str, (x, y), size, rgb, bold, anchor)] drawn with a dark outline. Returns a new array."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for s, xy, size, col, bold, anchor in items:
        dr.text(xy, s, font=font(size, bold), fill=tuple(col), anchor=anchor, stroke_width=max(2, size // 8),
                stroke_fill=(0, 0, 0))
    return np.array(im)


def banner(k):
    _, _, name, line = STAGES[k]
    b = np.full((H_BAN, W_OUT, 3), 28, np.uint8)
    b[-3:] = COLORS['A']
    return text(b, [(f'Stage {k} — {name}', (14, 8), 24, (255, 255, 255), True, 'la'),
                    (line, (14, 40), 16, (210, 210, 210), False, 'la'),
                    (f'{TPL} — event-anchored closest', (W_OUT - 14, 10), 16, (200, 200, 200), True, 'ra')])


def compose(k, body):
    assert body.shape == (H_BODY, W_OUT, 3), body.shape
    return np.vstack([banner(k), body])


def blank():
    return np.full((H_BODY, W_OUT, 3), 18, np.uint8)


def paste(canvas, img, x, y):
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


def fit(img, w):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR)


def cmap_lut(name):
    return (plt.get_cmap(name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


TURBO = cmap_lut('turbo')


def rnorm(r, lo, hi):
    """Log scale: near distances get most of the colour range."""
    return np.clip((np.log(r) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)


def r_color(r, lo, hi):
    return tuple(int(c) for c in TURBO[int(rnorm(r, lo, hi) * 255)])


def colorbar(h, lo, hi, title):
    """Vertical colorbar with meter ticks; returns (h, 90, 3)."""
    bar = np.full((h, 90, 3), 18, np.uint8)
    y0, y1 = 24, h - 10
    grad = TURBO[np.linspace(255, 0, y1 - y0).astype(int)]
    bar[y0:y1, 8:26] = grad[:, None]
    items = [(title, (4, 4), 13, (230, 230, 230), False, 'la')]
    for v in [x for x in (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 80) if lo <= x <= hi]:
        y = int(y1 - rnorm(v, lo, hi) * (y1 - y0 - 1))
        bar[y, 26:32] = 230
        items.append((f'{v:g}', (35, y), 13, (230, 230, 230), False, 'lm'))
    return text(bar, items)


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return a


def dark_axes(fig, axes):
    bg = (18 / 255,) * 3
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors='w', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color((0.5, 0.5, 0.5))
    return bg


def circle(img, u, v, R, col):
    c = (int(round(u)), int(round(v)))
    cv2.circle(img, c, R, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, c, R, col, 2, cv2.LINE_AA)


def trail(img, pts, col, thick=2):
    pts = np.round(pts).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)


def mcol(L):
    return np.array(COLORS[L]) / 255


def title_card(k):
    body = blank()
    body = text(body, [(f'Stage {k} — {STAGES[k][2]}', (W_OUT // 2, H_BODY // 2 - 20), 44, (255, 255, 255), True, 'mm'),
                       (STAGES[k][3], (W_OUT // 2, H_BODY // 2 + 36), 20, (210, 210, 210), False, 'mm')])
    return np.vstack([np.full((H_BAN, W_OUT, 3), 28, np.uint8), body])


def save_clip(k, frames, still):
    name = f'stage{k}_{STAGES[k][1]}'
    write_mp4(f'{OUT_I}/{name}.mp4', frames, FPS_OUT)
    Image.fromarray(frames[still]).save(f'{OUT_I}/stage{k}.png')
    print(f'{name}.mp4: {len(frames)} frames = {len(frames) / FPS_OUT:.1f} s, still = frame {still}', flush=True)
    return frames


def lidar_dots(img, t, lo, hi, rad=2):
    L = lidar(d, t)
    r = np.linalg.norm(L['Xc'], axis=1)
    for (u, v), rr in zip(L['uv'][::2], r[::2]):
        cv2.circle(img, (int(u), int(v)), rad, r_color(rr, lo, hi), -1)
    return img


def draw_boxes(img, t, col=(214, 39, 40)):
    for typ, p, (h, w, l) in boxes_at(d, t):
        Xc = xform(d['T_cam2_velo'], box_corners(p, h, w, l))
        if (Xc[:, 2] < 0.5).any():
            continue
        uv = (Xc @ K.T)[:, :2] / Xc[:, 2:3]
        uv = np.round(uv).astype(np.int32)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), col, 2, cv2.LINE_AA)
    return img


def track_pts(j, a, b):
    """On-screen pixel positions of t0 return j over frames a..b (for trails)."""
    return np.array([uvt[s][j] for s in range(a, b + 1) if ont[s][j]]).reshape(-1, 2)


def point_uv(L, t):
    return uvt[int(t)][P[L]]


def point_vis(L, t):
    return bool(vist[int(t)][P[L]])


def markers(img, t, R, labels=True, size=22):
    items = []
    for L in P:
        if not ont[int(t)][P[L]]:  # off-screen (e.g. passed by after t_e)
            continue
        u, v = point_uv(L, t)
        hidden = not point_vis(L, t)
        draw_marker(img, u, v, L, R, faded=hidden)
        if labels:
            left = u < W - 200
            items.append((f'{rt_[int(t)][P[L]]:.1f} m{" (hidden)" if hidden else ""}',
                          (int(u) + R + 4 if left else int(u) - R - 4, int(v) + R + 22 if v < H - 40 else int(v) - R - 8),
                          size, GRAY if hidden else COLORS[L], True, 'ls' if left else 'rs'))
    return items


# ---------------------------------------------------------------- the item
sel = json.load(open(f'{OUT}/selection.json'))
it = sel['picks'][args.pick]
OUT_I = f'{OUT}/item{args.pick}'
os.makedirs(OUT_I, exist_ok=True)
drive, EV, te, t0, tn = it['drive'], it['event'], it['te'], it['t0'], it['tn']
d = load(drive)
T, fps, H, W, K = d['n_frames'], d['fps'], d['H'], d['W'], d['K']
ev = events(d)
win = windows(d, ev)
frames_b = np.array(bake_indices(t0, tn, [t0, te, tn]))
k_e = int(np.nonzero(frames_b == te)[0][0])
Wb, Hb, sb = baked_size(H, W)
Rb = ring_radius(Wb)
R_DISP = ring_radius(W)
pad = marker_pad(d)
cand = candidates_t0(d, t0, te, frames_b)
P = dict(zip(LETTERS, it['j']))  # letter -> index into the t0 LiDAR returns
# every window frame: camera coordinates, pixel, visibility (z-buffer) and r of every t0 return
Xct, uvt, vist, rt_, ont = {}, {}, {}, {}, {}
for t in range(t0, tn + 1):
    Xct[t] = xform(velo_t0_to_cam(d, t0, t), cand['Xv'])
    vist[t], uvt[t], ont[t] = zbuffer(d, t, Xct[t])
    rt_[t] = np.linalg.norm(Xct[t], axis=1)
r_pt = {L: np.array([rt_[t][j] for t in range(t0, tn + 1)]) for L, j in P.items()}  # index = t - t0
closest_te = min(P, key=lambda L: r_pt[L][te - t0])
closest_t0 = min(P, key=lambda L: r_pt[L][0])
label = answer = closest_te
options = LETTERS
question = Q.format(event=EVENT_TEXT[EV])
margins_te = {f'{a}{b}': float(rel_margin(r_pt[a][te - t0], r_pt[b][te - t0])) for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]}
margins_t0 = {f'{a}{b}': float(rel_margin(r_pt[a][0], r_pt[b][0])) for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]}
print(drive, EV, 't_e', te, 'window', t0, tn, 'points', P, 'closest t0', closest_t0, 't_e', closest_te, flush=True)
r_lo, r_hi = 3.0, 80.0  # LiDAR colormap range (m, log)

# ---------------------------------------------------------------- stage 0
out = []
iw = 760
ih = int(round(H * iw / W))
cb = colorbar(2 * ih + 22, r_lo, r_hi, 'r (m, log)')
vf = d['velo_frames']
for t in vf:
    body = blank()
    img = d['image'](t)
    dots = np.full_like(img, 0)
    lidar_dots(dots, t, r_lo, r_hi, 2)
    y0_ = 24
    paste(body, fit(img, iw), 10, y0_)
    paste(body, fit(dots, iw), 10, y0_ + ih + 22)
    paste(body, cb, iw + 30, y0_)
    body = text(body, [('RGB (camera 2, rectified)', (10, y0_ - 6), 15, (230, 230, 230), True, 'ls'),
                       ('Velodyne returns in the image, r = ‖X_cam2‖ (Euclidean)', (10, y0_ + ih + 18), 15, (230, 230, 230), True, 'ls'),
                       (f'{drive}   frame {t}/{T - 1}   t = {t / fps:5.1f} s   speed {d["speed"][t]:.1f} m/s   '
                        f'yaw rate {d["yaw_rate"][t]:+.2f} rad/s', (10, H_BODY - 30), 13, (190, 190, 190), False, 'la'),
                       (f'K: f = {K[0, 0]:.0f} px, c = ({K[0, 2]:.0f}, {K[1, 2]:.0f}), {W}×{H}, {fps:g} fps;  '
                        f'pose per frame from OXTS (GPS/IMU);  velo→cam from calib', (10, H_BODY - 14), 13, (190, 190, 190), False, 'la')])
    out.append(compose(0, body))
clip0 = save_clip(0, out, int(np.argmin(np.abs(np.array(vf) - te))))

# ---------------------------------------------------------------- stage 1
rng = np.random.default_rng(0)
stat = np.nonzero(~cand['box'] & vist[t0] & (rt_[t0] < R_MAX))[0]
pick = rng.choice(stat, min(80, len(stat)), replace=False)
TR = int(round(fps))
out = []
seq1 = [t0] * 10 + list(range(t0, tn + 1))
iw1 = W_OUT - 20
for n_, t in enumerate(seq1):
    img = d['image'](t)
    for j in pick:
        lo = max(t0, t - TR)
        seg = track_pts(j, lo, t)
        trail(img, seg, r_color(rt_[t][j], r_lo, r_hi) if vist[t][j] else GRAY, 1)
    for j in pick:
        u, v = uvt[t][j]
        if not (0 <= u < W and 0 <= v < H):
            continue
        c = (int(round(u)), int(round(v)))
        if vist[t][j]:
            cv2.circle(img, c, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, 4, r_color(rt_[t][j], r_lo, r_hi), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(img, c, 5, GRAY, 2, cv2.LINE_AA)
    body = blank()
    im_ = fit(img, iw1)
    paste(body, im_, 10, 20)
    nv = int(vist[t][pick].sum())
    y_ = 20 + im_.shape[0] + 26
    body = text(body, [(f'{len(pick)} static LiDAR returns from t0 = frame {t0}, moved to frame {t} with the OXTS ego pose '
                        f'(t0 + {(t - t0) / fps:.1f} s)', (10, y_), 15, (230, 230, 230), True, 'la'),
                       (f'X_cam(t) = T_cam←velo · T_velo(t)⁻¹ · T_velo(t0) · X_velo(t0)       visible: {nv}   '
                        f'hidden / off-screen: {len(pick) - nv}', (10, y_ + 26), 14, (210, 210, 210), False, 'la'),
                       (f'visible = frame-t LiDAR has a return within {ZB_RAD} px at the same depth (±{ZB_TOL:.0%}) '
                        f'and none more than {ZB_TOL:.0%} closer (z-buffer)', (10, y_ + 50), 13, (190, 190, 190), False, 'la'),
                       ('filled = visible, colour = r (same scale as stage 0);  hollow gray = hidden;  trail = last 1 s',
                        (10, y_ + 72), 13, (190, 190, 190), False, 'la')])
    out.append(compose(1, body))
clip1 = save_clip(1, out, 10 + (te - t0))

# ---------------------------------------------------------------- stage 2
tt = np.arange(T) / fps
vid_w = 520
vid_i = list(range(0, T, max(1, T // 300)))
vid = {i: fit(d['image'](i), vid_w) for i in vid_i}
nwin = len(win)
per = max(1, nwin // 70)
steps = list(range(0, nwin, per)) + [nwin - 1]
EV_COL = {'stop': (1.0, 0.3, 0.3), 'turn_left': (0.3, 1.0, 0.3), 'turn_right': (0.8, 0.5, 1.0)}
out = []
A_COL = COLORS['A']


def stage2_frame(k_done, show_chosen):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.4, 2.5), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.17, hspace=0.12)
    dark_axes(fig, [ax, bx])
    ax.plot(tt, d['speed'], color=(0.2, 0.6, 1.0), lw=1.3)
    ax.axhline(STOP_V, color=(1.0, 0.3, 0.3), ls=':', lw=1)
    ax.axhline(STOP_FROM, color=(1.0, 0.3, 0.3), ls=':', lw=0.7)
    ax.set_ylabel('speed\n(m/s)', color='w', fontsize=8)
    bx.plot(tt, d['yaw_rate'], color=(1.0, 0.6, 0.2), lw=1.3)
    for s_ in (TURN_W, -TURN_W):
        bx.axhline(s_, color=(0.6, 0.6, 0.6), ls=':', lw=1)
    bx.set_ylabel('yaw rate\n(rad/s)', color='w', fontsize=8)
    bx.set_xlabel('time (s)', color='w', labelpad=1)
    w = win[min(k_done, nwin - 1)]
    for a_ in (ax, bx):
        a_.axvspan(w['t0'] / fps, w['tn'] / fps, color=(1, 1, 1), alpha=0.15)
        a_.axvspan(vf[0] / fps, vf[-1] / fps, color=(0.4, 0.4, 0.9), alpha=0.12, lw=0)
        for kind, e in ev:
            if e <= w['tn'] or show_chosen:
                a_.axvline(e / fps, color=EV_COL[kind], lw=1.2, ls='--')
        if show_chosen:
            a_.axvspan(t0 / fps, tn / fps, color=np.array(A_COL) / 255, alpha=0.35)
        a_.set_xlim(0, tt[-1])
    ax.text(vf[0] / fps, ax.get_ylim()[1] * 0.92, ' LiDAR fetched', color=(0.6, 0.6, 1.0), fontsize=7, va='top')
    return fig_to_array(fig)


x0, x1 = int(0.07 * 940), int(0.99 * 940)


def tx(t):
    return int(round(x0 + (x1 - x0) * t / tt[-1])) + 10


for k in steps + [nwin - 1] * 25:
    final = len(out) >= len(steps)
    w = win[k]
    body = blank()
    ci = w['t0'] if not final else te
    fi = vid[min(vid_i, key=lambda q: abs(q - ci))]
    paste(body, fi, (W_OUT - vid_w) // 2, 4)
    ytl = 4 + fi.shape[0] + 6
    cv2.rectangle(body, (tx(0), ytl), (tx(tt[-1]), ytl + 14), (60, 60, 60), -1)
    for q in range(k + 1):
        wq = win[q]
        cv2.rectangle(body, (tx(wq['t0'] / fps), ytl), (tx(min(wq['t0'] + int(round(STEP_SEC * fps)), T - 1) / fps), ytl + 14),
                      PASS_COL if wq['passed'] else FAIL_COL, -1)
    if not final:
        cv2.rectangle(body, (tx(w['t0'] / fps), ytl - 4), (tx(w['tn'] / fps), ytl + 18), (255, 255, 255), 2)
    else:
        cv2.rectangle(body, (tx(t0 / fps), ytl - 5), (tx(tn / fps), ytl + 19), A_COL, 3)
    plot = stage2_frame(k, final)
    paste(body, plot, 10, ytl + 20)
    npass = sum(wq['passed'] for wq in win[:k + 1])
    why = '' if w['passed'] else ('  (no event ≥ 1 s from both ends)' if not w['events'] else '  (no LiDAR on disk)')
    evs = ', '.join(f'{k_} @ {e / fps:.1f} s' for k_, e in w['events'])
    msg = (f'window {k + 1}/{nwin}: t = {w["t0"] / fps:.1f}–{w["tn"] / fps:.1f} s   {evs}   → {"PASS" if w["passed"] else "FAIL"}{why}'
           if not final else
           f'chosen: {EV} at frame {te} (t = {te / fps:.1f} s); window frames {t0}–{tn} ({(te - t0) / fps:.1f} s before, '
           f'{(tn - te) / fps:.1f} s after)')
    body = text(body, [(msg, (W_OUT // 2, ytl + 20 + plot.shape[0] + 10), 13,
                        A_COL if final else (PASS_COL if w['passed'] else FAIL_COL), True, 'mm'),
                       (f'stop: speed < {STOP_V} m/s after > {STOP_FROM:g} m/s within {STOP_LOOK:g} s;  turn: |yaw rate| crosses '
                        f'{TURN_W} rad/s and holds ≥ {TURN_HOLD} s (+ = left);  {N_SEC} s windows, stride {STEP_SEC} s',
                        (W_OUT // 2, ytl + 20 + plot.shape[0] + 28), 11, (180, 180, 180), False, 'mm')])
    out.append(compose(2, body))
clip2 = save_clip(2, out, len(out) - 1)

# ---------------------------------------------------------------- stage 3
out = []
L0 = lidar(d, t0)
n_box = int(cand['box'].sum())
types0 = sorted(set(b[0] for b in boxes_at(d, t0)))
for q, t in enumerate([t0] * 12 + list(range(t0, tn + 1))):
    img = d['image'](t)
    img = (img * 0.6).astype(np.uint8)
    on = (q < 12) | (vist[t] & ~cand['box'])  # after t0 only the static returns (movers are not ego-propagated)
    for j in np.nonzero(on)[0][::2]:
        u, v = uvt[t][j]
        if not (0 <= u < W and 0 <= v < H):
            continue
        col = (214, 39, 40) if cand['box'][j] else r_color(rt_[t][j], r_lo, r_hi)
        cv2.circle(img, (int(u), int(v)), 2, col, -1)
    draw_boxes(img, t)
    body = blank()
    im_ = fit(img, W_OUT - 20)
    paste(body, im_, 10, 24)
    y_ = 24 + im_.shape[0] + 26
    body = text(body, [(f'frame {t}  (t0 + {(t - t0) / fps:.1f} s)', (10, 16), 14, (230, 230, 230), False, 'la'),
                       (f'kept: {int((~cand["box"]).sum())} static LiDAR returns at t0 (coloured by r)', (10, y_), 16,
                        (120, 230, 120), True, 'la'),
                       (f'dropped: {n_box} returns inside {len(boxes_at(d, t0))} tracklet boxes at t0 '
                        f'({", ".join(types0) or "none"}; boxes grown by {BOX_PAD} m) — red', (10, y_ + 26), 14,
                        (230, 120, 120), False, 'la'),
                       ('KITTI has no instance masks: a "region" is the static background, i.e. every LiDAR return outside the '
                        'annotated movers', (10, y_ + 52), 13, (190, 190, 190), False, 'la'),
                       ('after t0 the kept returns are moved with the ego pose (stage 1); red returns would move on their own',
                        (10, y_ + 72), 13, (190, 190, 190), False, 'la')])
    out.append(compose(3, body))
clip3 = save_clip(3, out, 5)

# ---------------------------------------------------------------- stage 4
# Rejected examples: other returns tried in place of C (A and B fixed), first failing check in pipeline order.
first_ok = np.stack([vist[int(t)] for t in frames_b[:N_MARKED]]).all(0)
marked_in = np.stack([in_bounds(uvt[int(t)], W, H, pad) for t in frames_b[:N_MARKED]]).all(0)
pool = np.nonzero(~cand['box'] & (rt_[t0] <= R_MAX) & marked_in)[0]
pool = pool[~np.isin(pool, list(P.values()))]
rng = np.random.default_rng(1)
rng.shuffle(pool)
fails = {'hidden at t0 / marked frames': [], 'hidden at t_e': [], 'depth variance': [], 'round trip > 3 px': [],
         'too close to a marker': [], f'margin at t_e < {M:.0%}': []}
for j in pool[:5000]:
    near = min(np.linalg.norm(np.array([uvt[int(t)][j] - uvt[int(t)][P[L]] for t in frames_b[:N_MARKED]]), axis=-1).min()
               for L in ('A', 'B'))
    if not first_ok[j]:
        fails['hidden at t0 / marked frames'].append((int(j), 'z-buffer: occluded / no return'))
        continue
    if not vist[te][j]:
        fails['hidden at t_e'].append((int(j), 'off-screen at t_e' if not in_bounds(uvt[te][j:j + 1], W, H)[0] else 'occluded at t_e'))
        continue
    if cand['patch'][j] >= STD_MAX:
        if np.isfinite(cand['patch'][j]):
            fails['depth variance'].append((int(j), f'σ/μ = {cand["patch"][j]:.0%}'))
        continue
    if cand['rt_err'][j] > RT_MAX:
        fails['round trip > 3 px'].append((int(j), 'no matching return' if not np.isfinite(cand['rt_err'][j]) else f'{cand["rt_err"][j]:.1f} px'))
        continue
    if near < pad:
        fails['too close to a marker'].append((int(j), f'{near:.0f} px < {pad:.0f} px'))
        continue
    mm = min(rel_margin(rt_[te][j], r_pt[L][te - t0]) for L in ('A', 'B'))
    if mm < M:
        fails[f'margin at t_e < {M:.0%}'].append((int(j), f'min margin = {mm:.0%}'))
examples = {k: [] for k in fails}
placed = [uvt[t0][j] for j in P.values()]
for why in sorted(fails, key=lambda k: len(fails[k])):
    for j, note in fails[why]:
        if len(examples[why]) >= 2:
            break
        u, v = uvt[t0][j]
        if any(abs(u - p[0]) < 230 and abs(v - p[1]) < 40 for p in placed):
            continue
        if not (40 < v < H - 20 and u < W - 240):
            continue
        examples[why].append((j, note))
        placed.append(uvt[t0][j])
print('stage 4 rejected examples:', examples, flush=True)
missing4 = [k for k, v in examples.items() if not v]
spacing = float(min(np.linalg.norm(np.array([uvt[int(t)][P[a]] - uvt[int(t)][P[b]] for t in frames_b[:N_MARKED]]), axis=-1).min()
                    for ai, a in enumerate(LETTERS) for b in LETTERS[ai + 1:]))
out = []
for q in [0] * 20 + list(range(0, tn - t0 + 1)) + [tn - t0] * 5:
    t = t0 + q
    img = d['image'](t)
    items = []
    if q == 0 and len(out) < 20:
        for why, ex in examples.items():
            for j, note in ex:
                u, v = uvt[t0][j]
                circle(img, u, v, 9, GRAY)
                cv2.line(img, (int(u) - 6, int(v) - 6), (int(u) + 6, int(v) + 6), GRAY, 2, cv2.LINE_AA)
                items.append((f'✗ {why}', (int(u) + 13, int(v) - 3), 15, (215, 215, 215), True, 'ls'))
                items.append((note, (int(u) + 13, int(v) + 15), 14, (190, 190, 190), False, 'ls'))
        for L, j in P.items():
            draw_marker(img, *uvt[t0][j], L, R_DISP)
        lines = [('✓ chosen points pass all checks', (120, 230, 120), True)]
        for L, j in P.items():
            lines.append((f'{L}: static LiDAR return, r(t0) = {r_pt[L][0]:.1f} m, r(t_e) = {r_pt[L][te - t0]:.1f} m, '
                          f'5×5 σ/μ = {cand["patch"][j]:.1%}, round trip {cand["rt_err"][j]:.1f} px', COLORS[L], False))
        lines.append((f'margins at t_e: ' + ', '.join(f'{k} {v:.0%}' for k, v in margins_te.items()) +
                      f';  min marker spacing {spacing:.0f} px ≥ {pad:.0f} px', (230, 230, 230), False))
        caption = 'Candidate static LiDAR returns at t0 (gray = rejected candidates for C)'
    else:
        for L, j in P.items():
            trail(img, track_pts(j, t0, t), COLORS[L], 2)
        items += markers(img, t, R_DISP, labels=False)
        caption = f'propagated tracks   t0 + {q / fps:.1f} s{"   ← t_e (" + EV + ")" if t == te else ""}   (gray ring = hidden by z-buffer)'
        lines = []
    body = blank()
    img = text(img, items)
    im_ = fit(img, W_OUT - 20)
    paste(body, im_, 10, 28)
    y_ = 28 + im_.shape[0] + 22
    bi = [(caption, (10, 18), 15, (240, 240, 240), True, 'ls')]
    bi += [(s, (10, y_ + 22 * n), 14, col, bold, 'la') for n, (s, col, bold) in enumerate(lines)]
    bi.append((f'checks: visible (z-buffer) on the first {N_MARKED} frames and at t_e;  LiDAR σ/μ in 5×5 < {STD_MAX:.0%};  '
               f'round trip t0→t_e→t0 ≤ {RT_MAX:g} px;', (10, H_BODY - 28), 12, (200, 200, 200), False, 'ls'))
    bi.append((f'markers ≥ 3 ring radii apart;  pairwise margin ≥ {M:.0%} at t_e', (10, H_BODY - 10), 12, (200, 200, 200), False, 'ls'))
    out.append(compose(4, text(body, bi)))
clip4 = save_clip(4, out, 5)

# ---------------------------------------------------------------- stage 5
out = []
tw = np.arange(tn - t0 + 1) / fps
allr = np.concatenate(list(r_pt.values()))


def stage5_plot(q):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.6, 2.35), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    fig.subplots_adjust(left=0.12, right=0.84, top=0.9, bottom=0.2, hspace=0.1)
    dark_axes(fig, [ax, bx])
    for x, s in [(0, 't0'), ((te - t0) / fps, 't_e')]:
        ax.axvline(x, color='w' if s == 't0' else (0.3, 1.0, 0.3), ls=':', lw=1.2)
        bx.axvline(x, color='w' if s == 't0' else (0.3, 1.0, 0.3), ls=':', lw=1.2)
    for L in P:
        ax.plot(tw[:q + 1], r_pt[L][:q + 1], color=mcol(L), lw=2)
        ax.plot(tw[q], r_pt[L][q], 'o', color=mcol(L), ms=5, mec='k')
    if q >= te - t0:
        for L in P:
            ax.text(tw[-1] + 0.1, r_pt[L][te - t0], f'{L} {r_pt[L][te - t0]:.1f} m', color=mcol(L), fontsize=8,
                    weight='bold' if L == closest_te else 'normal', va='center')
    ax.set_ylim(allr.min() * 0.85, allr.max() * 1.1)
    ax.set_ylabel('r (m)', color='w', fontsize=8)
    ax.set_title('GT distance of each point', color='w', fontsize=9)
    bx.plot(tw[:q + 1], d['speed'][t0:t0 + q + 1], color=(0.2, 0.6, 1.0), lw=1.3, label='speed (m/s)')
    bx.plot(tw[:q + 1], 10 * d['yaw_rate'][t0:t0 + q + 1], color=(1.0, 0.6, 0.2), lw=1.3, label='10 × yaw rate')
    bx.set_ylim(min(-2, 10 * d['yaw_rate'][t0:tn + 1].min() - 1), max(d['speed'][t0:tn + 1].max(), 10 * d['yaw_rate'][t0:tn + 1].max()) + 1)
    bx.legend(loc='upper left', fontsize=6, facecolor=(0.1, 0.1, 0.1), labelcolor='w', framealpha=0.8, ncol=2)
    bx.set_xlim(-0.1, tw[-1] + 0.1)
    bx.set_xlabel('time since t0 (s)', color='w', fontsize=8, labelpad=1)
    return fig_to_array(fig)


seq5 = list(range(tn - t0 + 1))
seq5 = seq5[:te - t0] + [te - t0] * 15 + seq5[te - t0:] + [tn - t0] * 15
for q in seq5:
    t = t0 + q
    img = d['image'](t)
    for L, j in P.items():
        trail(img, track_pts(j, t0, t), COLORS[L], 2)
    img = text(img, markers(img, t, R_DISP, True, 26))
    if t == te:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (100, 240, 100), 8)
    body = blank()
    im_ = fit(img, 900)
    paste(body, im_, 30, 20)
    plot = stage5_plot(q)
    y_ = 20 + im_.shape[0] + 4
    paste(body, plot, 4, y_)
    cur = min(P, key=lambda L: r_pt[L][q])
    x_ = plot.shape[1] + 14
    items = [(f'frame {t}   t0 + {q / fps:.1f} s' + ('   = t_e' if t == te else ''), (10, 16), 14, (230, 230, 230), False, 'ls'),
             (f'closest at t0: {closest_t0}', (x_, y_ + 18), 16, (200, 200, 200), True, 'la'),
             (f'closest now:  {cur}', (x_, y_ + 44), 16, COLORS[cur], True, 'la')]
    if q >= te - t0:
        items += [(f'at t_e ({EVENT_TEXT[EV]}):', (x_, y_ + 78), 14, (100, 240, 100), False, 'la'),
                  (f'closest = {closest_te}', (x_, y_ + 100), 22, (100, 240, 100), True, 'la'),
                  ('margins ' + ', '.join(f'{k} {v:.0%}' for k, v in margins_te.items()), (x_, y_ + 132), 12, (200, 200, 200), False, 'la'),
                  (f'(all ≥ {M:.0%}); ' + ('differs from t0' if closest_te != closest_t0 else 'same as at t0'),
                   (x_, y_ + 150), 12, (200, 200, 200), False, 'la')]
    out.append(compose(5, text(body, items)))
clip5 = save_clip(5, out, len(out) - 1)

# ---------------------------------------------------------------- stage 6
baked = []
for k, t in enumerate(frames_b):  # identical to render.py: INTER_AREA resize, markers on the first N_MARKED frames
    img = cv2.resize(d['image'](int(t)), (Wb, Hb), interpolation=cv2.INTER_AREA)
    if k < N_MARKED:
        for L in P:
            u, v = point_uv(L, t)
            draw_marker(img, u * sb, v * sb, L, Rb)
    baked.append(img)
in_fps = round((len(frames_b) - 1) / ((frames_b[-1] - frames_b[0]) / fps), 3)
write_mp4(f'{OUT_I}/stage6_model_input.mp4', baked, in_fps)
out = []
for k, img in enumerate(baked):
    body = blank()
    disp = fit(img, W_OUT)
    paste(body, disp, 0, 70)
    body = text(body, [(f'frame {k + 1:2d} / {len(baked)}', (W_OUT - 12, 12), 20, (255, 255, 255), True, 'ra'),
                       (f'source frame {frames_b[k]}   {"marked" if k < N_MARKED else "no marker"}'
                        f'{"   (t_e, forced)" if k == k_e else ""}', (W_OUT - 12, 40), 13,
                        A_COL if k < N_MARKED else (100, 240, 100) if k == k_e else (190, 190, 190), False, 'ra'),
                       (f'model input: {Wb}×{Hb}, {len(baked)} frames at {in_fps:g} fps (shown at 5 fps, downscaled '
                        f'for display); t_e = frame {te} is always baked', (10, H_BODY - 16), 13, (230, 230, 230), False, 'ls')])
    out += [compose(6, body)] * 2
clip6 = save_clip(6, out, 0)

# ---------------------------------------------------------------- stage 7
body = blank()
thumb = fit(baked[0], 560)
paste(body, thumb, W_OUT - 560 - 16, 90)
items = [('first input frame (marked)', (W_OUT - 16 - 280, 90 + thumb.shape[0] + 20), 13, (190, 190, 190), False, 'mm'),
         ('Question', (24, 60), 16, (170, 170, 170), True, 'la')]
y = 88
for line in textwrap.wrap(question, 30):
    items.append((line, (24, y), 21, (255, 255, 255), False, 'la'))
    y += 30
y += 20
items.append(('Options', (24, y), 16, (170, 170, 170), True, 'la'))
y += 28
for o in options:
    ok_ = o == answer
    items.append((('● ' if ok_ else '○ ') + o, (40, y), 21, (100, 240, 100) if ok_ else (220, 220, 220), ok_, 'la'))
    y += 30
y += 20
items.append(('Answer', (24, y), 16, (170, 170, 170), True, 'la'))
items.append((answer, (24, y + 26), 30, (100, 240, 100), True, 'la'))
items.append(('GT at t_e: ' + ', '.join(f'r_{L} = {r_pt[L][te - t0]:.1f} m' for L in sorted(P, key=lambda L: r_pt[L][te - t0])) +
              f';  closest at t0: {closest_t0}', (24, H_BODY - 44), 13, (170, 170, 170), False, 'la'))
items.append((f'{drive}, frames {t0}–{tn}, t_e = {te} ({EV}, from OXTS)', (24, H_BODY - 24), 13, (170, 170, 170), False, 'la'))
card = compose(7, text(body, items))
clip7 = save_clip(7, [card] * 50, 0)

# ---------------------------------------------------------------- full video, meta, checks
full = []
for k, c in enumerate([clip0, clip1, clip2, clip3, clip4, clip5, clip6, clip7]):
    full += [title_card(k)] * int(1.5 * FPS_OUT) + c
write_mp4(f'{OUT_I}/walkthrough_full.mp4', full, FPS_OUT)
print(f'walkthrough_full.mp4: {len(full) / FPS_OUT:.1f} s', flush=True)

# consistency checks: at t_e, the nearest frame-t_e LiDAR return to each marker gives its own r (label recomputed
# from the t_e scan, not the propagated t0 points); z-buffer visibility; markers only on the first 3 frames
Le = lidar(d, te)
r_scan = {}
for L, j in P.items():
    dd_, ii = Le['tree'].query(uvt[te][j], k=1)
    r_scan[L] = float(np.linalg.norm(Le['Xc'][ii]))
label_scan = min(r_scan, key=r_scan.get)
ring_px = []
for k, t in enumerate(frames_b):
    plain = cv2.resize(d['image'](int(t)), (Wb, Hb), interpolation=cv2.INTER_AREA)
    ring_px.append(int((np.abs(baked[k].astype(int) - plain).sum(-1) > 30).sum()))
checks = dict(visible_t0={L: point_vis(L, t0) for L in P}, visible_te={L: point_vis(L, te) for L in P},
              visible_marked={L: all(point_vis(L, t) for t in frames_b[:N_MARKED]) for L in P},
              round_trip_px={L: float(cand['rt_err'][j]) for L, j in P.items()},
              r_te_from_te_scan=r_scan, label_from_te_scan=label_scan, label_consistent=label_scan == label,
              ring_changed_px_per_baked_frame=ring_px,
              ring_only_on_first_3=bool(all(p > 0 for p in ring_px[:N_MARKED]) and not any(ring_px[N_MARKED:])),
              t_e_baked=bool(te in frames_b))
print(json.dumps(checks), flush=True)

meta = dict(dataset='kitti_raw', video_id=drive, fps=fps, resolution=[W, H], template=TPL, event=EV, event_text=EVENT_TEXT[EV],
            t_e=int(te), window=[int(t0), int(tn)], t0=int(t0), tn=int(tn), t0_sec=t0 / fps, te_sec=te / fps, tn_sec=tn / fps,
            speed_te=float(d['speed'][te]), yaw_rate_te=float(d['yaw_rate'][te]),
            points={L: dict(lidar_index_t0=int(j), velo_xyz_t0=[float(x) for x in cand['Xv'][j]],
                            pixel_t0=[float(x) for x in uvt[t0][j]], r_t0=float(r_pt[L][0]), r_te=float(r_pt[L][te - t0]),
                            patch_rel_std_t0=float(cand['patch'][j]), round_trip_px=float(cand['rt_err'][j]),
                            r_baked=[round(float(rt_[int(t)][j]), 3) for t in frames_b],
                            vis_baked=[bool(vist[int(t)][j]) for t in frames_b]) for L, j in P.items()},
            closest_t0=closest_t0, closest_te=closest_te, changed=closest_te != closest_t0,
            margins_te=margins_te, margins_t0=margins_t0,
            label=label, question=question, options=options, answer=answer,
            baked_frames=[int(i) for i in frames_b], baked_te_index=k_e + 1, baked_resolution=[Wb, Hb], baked_fps=in_fps,
            ring_radius_px=Rb, marker_spacing_px_min=spacing, events_in_drive=[[k_, int(e)] for k_, e in ev],
            thresholds=dict(m=M, clarity_min_margin=M_SHOW, stop_speed=STOP_V, stop_from_speed=STOP_FROM,
                            stop_lookback_s=STOP_LOOK, turn_yaw_rate=TURN_W, turn_hold_s=TURN_HOLD, zbuffer_rel_tol=ZB_TOL,
                            zbuffer_radius_px=ZB_RAD, round_trip_px=RT_MAX, patch_std=STD_MAX, box_pad_m=BOX_PAD,
                            r_max_m=R_MAX, window_s=N_SEC, window_stride_s=STEP_SEC),
            windows=dict(n=len(win), passed=int(sum(w['passed'] for w in win))),
            stage3=dict(returns_t0=int(len(cand['Xv'])), in_boxes=n_box, box_types=types0),
            stage4_rejected={k: [j for j, _ in v] for k, v in examples.items()}, stage4_missing_reasons=missing4,
            checks=checks)
json.dump(meta, open(f'{OUT_I}/meta.json', 'w'), indent=1)
