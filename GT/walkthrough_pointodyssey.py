# Stage-by-stage walkthrough clips for the point-level depth video QA pipeline (PointOdyssey, T1).
# One video, one 2 s window, one point A across stages 0-7; every number comes from PointOdyssey GT
# (depth, camera, 3D tracks, visibility, instance masks). Writes to <OUT_ROOT>/walkthrough/:
#   stage{k}_<name>.mp4, stage{k}.png, walkthrough_full.mp4, meta.json, README.md
#   python walkthrough_pointodyssey.py --select           # scan test scenes, write selection.json
#   python walkthrough_pointodyssey.py                    # render the top-ranked item from selection.json
#   python walkthrough_pointodyssey.py --pick 3           # render the 4th-ranked item instead
import argparse, json, os, sys, textwrap
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blk import PO
from common import OUT_ROOT, bake_indices, euclid, in_bounds, load_po, patch_rel_std, po_scenes
from render import COLORS, N_MARKED, baked_size, draw_marker, ring_radius, write_mp4

OUT = f'{OUT_ROOT}/walkthrough'
FPS_OUT = 10
W_OUT, H_BAN, H_BODY = 960, 64, 540
N_SEC = 2
M = 0.10  # T1 relative margin
M_SHOW = 0.25  # the walkthrough item needs a much larger change, for visual clarity
STD_MAX = 0.02  # relative z std in a 5x5 patch
DEPTH_AGREE = 0.02  # |depth map z - track z| / z
STEP_SEC = 0.5  # sliding-window stride
CAM_MIN = 0.2  # m/s, mean camera speed over the window
AGENT_MIN = 0.5  # m/s, mean over the window of the fastest visible moving track
DYN_MIN = 0.05  # m, a track is "on an agent" if its world position ever moves more than this
REG_MIN_AREA = 0.004  # fraction of the image (median over the window)
REG_MIN_PRESENT = 0.9  # fraction of window frames in which the region is visible at all
REG_MAX_BORDER = 0.5  # fraction of window frames in which the region touches the image border
REG_BG_AREA = 0.25  # larger than this at t0 -> scene background, not an object region

A_COL = COLORS['A']  # (255, 220, 0) everywhere
GRAY = (150, 150, 150)
PASS_COL, FAIL_COL = (44, 160, 44), (214, 39, 40)
REGION_COLS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
               (23, 190, 207), (227, 119, 194), (140, 86, 75)]
FONT_DIR = f'{matplotlib.get_data_path()}/fonts/ttf'
STAGES = [(0, 'standardize', 'Standardize', 'Common format: RGB, intrinsics, poses, per-pixel distance to camera.'),
          (1, 'geometry', 'Geometry', '3D point tracks + visibility for every frame.'),
          (2, 'clips', 'Clips', 'Slide windows; keep those with enough motion.'),
          (3, 'regions', 'Regions', 'Pick valid regions from instance masks.'),
          (4, 'points', 'Points', 'Place the query point; keep it only if it passes the checks.'),
          (5, 'labels', 'Labels', 'Compute the answer from GT distance.'),
          (6, 'render', 'Render', 'Bake frames; mark the point on the first 3 frames only.'),
          (7, 'question', 'Question', 'Final QA item.')]
Q = 'Point A is marked at the start. After {n} seconds, is it closer to or farther from the camera than at the start?'

ap = argparse.ArgumentParser()
ap.add_argument('--select', action='store_true')
ap.add_argument('--scenes', default='', help='comma list; default all test scenes')
ap.add_argument('--pick', type=int, default=0)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- data + geometry
def load(scene):
    d = load_po(scene)
    a = np.load(f'{PO}/{scene}/anno.npz')
    E = a['extrinsics'].astype(np.float64)
    d['X'] = a['trajs_3d'].astype(np.float32)  # world frame
    d['C'] = np.array([-e[:3, :3].T @ e[:3, 3] for e in E])  # camera centers, world
    d['r'] = euclid(d['P_cam'])
    Ki = np.linalg.inv(d['K'])
    uu, vv = np.meshgrid(np.arange(d['W']), np.arange(d['H']))
    d['ray_norm'] = np.linalg.norm(np.stack([uu, vv, np.ones_like(uu)], -1) @ Ki.T, axis=-1).astype(np.float32)
    return d


def dist_map(d, i):
    z = d['depth'](i)
    r = z * d['ray_norm']  # r = z * ||K^-1 [u, v, 1]||
    r[(z <= 0) | (z >= 999)] = np.nan
    return r


def speed(P, fps, h=6):
    """Central-difference speed (m/s) over +-h frames for (T, ...) positions."""
    T = len(P)
    lo, hi = np.maximum(np.arange(T) - h, 0), np.minimum(np.arange(T) + h, T - 1)
    return np.linalg.norm(P[hi] - P[lo], axis=-1) / ((hi - lo) / fps).reshape((-1,) + (1,) * (P.ndim - 2))


def motion_summary(d):
    fps = d['fps']
    cam = speed(d['C'], fps)
    dyn = np.linalg.norm(d['X'] - d['X'][:1], axis=-1).max(0) > DYN_MIN
    S = speed(d['X'][:, dyn], fps)
    S[~d['vis'][:, dyn]] = 0
    agent = S.max(1) if dyn.any() else np.zeros(len(cam))
    return cam, agent, dyn


def windows(d, cam, agent):
    L = int(round(N_SEC * d['fps']))
    step = int(round(STEP_SEC * d['fps']))
    out = []
    for t0 in range(0, d['n_frames'] - L, step):
        tn = t0 + L
        ok = d['vis'][t0:t0 + N_MARKED].all(0) & d['vis'][tn]
        rel = (d['r'][tn] - d['r'][t0]) / d['r'][t0]
        c, g = float(cam[t0:tn + 1].mean()), float(agent[t0:tn + 1].mean())
        t1 = bool((ok & (np.abs(rel) >= M)).any())
        out.append(dict(t0=t0, tn=tn, cam=c, agent=g, motion=bool(c >= CAM_MIN or g >= AGENT_MIN), t1=t1,
                        passed=bool((c >= CAM_MIN or g >= AGENT_MIN) and t1)))
    return out


def regions(d, t0, tn):
    """Instance regions of the window. Returns {id: dict(keep, reason, color)} and the masks."""
    masks = [d['mask'](i) for i in range(t0, tn + 1)]
    H, W = masks[0].shape
    ids = [int(i) for i in np.unique(masks[0])]
    reg = {}
    for i in ids:
        area = np.array([(m == i).mean() for m in masks])
        border = np.array([(m[0] == i).any() or (m[-1] == i).any() or (m[:, 0] == i).any() or (m[:, -1] == i).any()
                           for m in masks])
        if area[0] > REG_BG_AREA:
            reason = 'background'
        elif (area > 0).mean() < REG_MIN_PRESENT:
            reason = 'not visible enough'
        elif np.median(area) < REG_MIN_AREA:
            reason = 'too small'
        elif border.mean() > REG_MAX_BORDER:
            reason = 'truncated'
        else:
            reason = ''
        reg[i] = dict(keep=reason == '', reason=reason, median_area=float(np.median(area)),
                      present=float((area > 0).mean()), border=float(border.mean()))
    k = 0
    for i in sorted(reg, key=lambda i: -reg[i]['median_area']):
        if reg[i]['keep']:
            reg[i]['color'] = REGION_COLS[k % len(REGION_COLS)]
            k += 1
    return reg, masks


def track_ids(d, t, mask):
    uv = np.round(d['uv'][t]).astype(int)
    ok = d['vis'][t] & in_bounds(uv, d['W'], d['H'])
    ids = np.full(len(uv), -1)
    ids[ok] = mask[uv[ok, 1], uv[ok, 0]]
    return ids


def depth_agree(dep, d, t, j):
    u, v = np.round(d['uv'][t, j]).astype(int)
    z = d['P_cam'][t, j, 2]
    return abs(dep[v, u] - z) / z


def marker_pad(d):
    Wb, Hb, s = baked_size(d['H'], d['W'])
    return 3 * ring_radius(Wb) / s


# ---------------------------------------------------------------- selection
if args.select:
    scenes = args.scenes.split(',') if args.scenes else [s for s in po_scenes() if s.startswith('test/')]
    cands = []
    for sc in scenes:
        d = load(sc)
        cam, agent, dyn = motion_summary(d)
        win = windows(d, cam, agent)
        pad = marker_pad(d)
        print(sc, d['n_frames'], 'cam p50/p90 %.2f/%.2f' % tuple(np.percentile(cam, [50, 90])),
              'agent p50/p90 %.2f/%.2f' % tuple(np.percentile(agent, [50, 90])),
              'windows pass %d/%d' % (sum(w['passed'] for w in win), len(win)), flush=True)
        sc_c = []
        for w in win:
            if not w['passed']:
                continue
            t0, tn = w['t0'], w['tn']
            rel = (d['r'][tn] - d['r'][t0]) / d['r'][t0]
            moved = np.linalg.norm(d['X'][tn] - d['X'][t0], axis=-1) > 0.1
            ok = dyn & moved & d['vis'][t0:tn + 1].all(0) & in_bounds(d['uv'][t0:tn + 1], d['W'], d['H'], pad).all(0) \
                & (np.abs(rel) >= M_SHOW)
            for j in np.nonzero(ok)[0]:
                cdist = np.hypot(*(d['uv'][t0, j] - [d['W'] / 2, d['H'] / 2])) / np.hypot(d['W'] / 2, d['H'] / 2)
                score = min(abs(rel[j]), 0.8) + 0.3 * (w['cam'] >= CAM_MIN) + 0.2 * (w['agent'] >= AGENT_MIN) \
                    + 0.2 * (1 - cdist)
                sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=int(j), rel=float(rel[j]), cam=w['cam'], agent=w['agent'],
                                 score=float(score)))
        # verify the best few per scene: patch std + depth agreement at t0 and tn, A inside a kept region
        sc_c.sort(key=lambda c: -c['score'])
        seen, kept = set(), 0
        for c in sc_c:
            if kept >= 3 or len(seen) >= 12:
                break
            if c['t0'] in seen:
                continue
            seen.add(c['t0'])
            dep0, depn = d['depth'](c['t0']), d['depth'](c['tn'])
            u, v = d['uv'][c['t0'], c['j']]
            if patch_rel_std(dep0, u, v) >= STD_MAX or depth_agree(dep0, d, c['t0'], c['j']) > DEPTH_AGREE \
                    or depth_agree(depn, d, c['tn'], c['j']) > DEPTH_AGREE:
                continue
            # clarity: A must be unoccluded in the depth map on every window frame, not only per the GT flag
            if max(depth_agree(d['depth'](i), d, i, c['j']) for i in range(c['t0'], c['tn'] + 1)) > DEPTH_AGREE:
                continue
            reg, masks = regions(d, c['t0'], c['tn'])
            rid = int(track_ids(d, c['t0'], masks[0])[c['j']])
            if rid not in reg or not reg[rid]['keep']:
                continue
            c['region'] = rid
            cands.append(c)
            kept += 1
            print('  cand', json.dumps(c), flush=True)
        del d
    cands.sort(key=lambda c: -c['score'])
    json.dump(cands, open(f'{OUT}/selection.json', 'w'), indent=1)
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
    b[-3:] = A_COL
    return text(b, [(f'Stage {k} — {name}', (14, 8), 24, (255, 255, 255), True, 'la'),
                    (line, (14, 40), 16, (210, 210, 210), False, 'la')])


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


def colorize(r, lo, hi):
    x = rnorm(r, lo, hi)
    out = TURBO[np.nan_to_num(x * 255).astype(np.uint8)]
    out[np.isnan(r)] = 0
    return out


def r_color(r, lo, hi):
    return tuple(int(c) for c in TURBO[int(rnorm(r, lo, hi) * 255)])


def colorbar(h, lo, hi, title):
    """Vertical colorbar with meter ticks; returns (h, 90, 3)."""
    bar = np.full((h, 90, 3), 18, np.uint8)
    y0, y1 = 24, h - 10
    grad = TURBO[np.linspace(255, 0, y1 - y0).astype(int)]
    bar[y0:y1, 8:26] = grad[:, None]
    items = [(title, (4, 4), 13, (230, 230, 230), False, 'la')]
    for v in [x for x in (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100) if lo <= x <= hi]:
        y = int(y1 - rnorm(v, lo, hi) * (y1 - y0 - 1))
        bar[y, 26:32] = 230
        items.append((f'{v:g}', (35, y), 13, (230, 230, 230), False, 'lm'))
    return text(bar, items)


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return a


def circle(img, u, v, R, col):
    c = (int(round(u)), int(round(v)))
    cv2.circle(img, c, R, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, c, R, col, 2, cv2.LINE_AA)


def trail(img, pts, col, thick=2):
    pts = np.round(pts).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)


def title_card(k):
    body = blank()
    body = text(body, [(f'Stage {k} — {STAGES[k][2]}', (W_OUT // 2, H_BODY // 2 - 20), 44, (255, 255, 255), True, 'mm'),
                       (STAGES[k][3], (W_OUT // 2, H_BODY // 2 + 36), 20, (210, 210, 210), False, 'mm')])
    return np.vstack([np.full((H_BAN, W_OUT, 3), 28, np.uint8), body])


def save_clip(k, frames, still):
    name = f'stage{k}_{STAGES[k][1]}'
    write_mp4(f'{OUT}/{name}.mp4', frames, FPS_OUT)
    Image.fromarray(frames[still]).save(f'{OUT}/stage{k}.png')
    print(f'{name}.mp4: {len(frames)} frames = {len(frames) / FPS_OUT:.1f} s, still = frame {still}', flush=True)
    return frames


# ---------------------------------------------------------------- the item
sel = json.load(open(f'{OUT}/selection.json'))
it = sel[args.pick]
scene, t0, tn, jA = it['scene'], it['t0'], it['tn'], it['j']
d = load(scene)
T, fps, H, W, K = d['n_frames'], d['fps'], d['H'], d['W'], d['K']
cam, agent, dyn = motion_summary(d)
win = windows(d, cam, agent)
reg, wmasks = regions(d, t0, tn)
rA = d['r'][:, jA]
labA = 'Closer' if rA[tn] < rA[t0] else 'Farther'
relA = (rA[tn] - rA[t0]) / rA[t0]
dep0, depn = d['depth'](t0), d['depth'](tn)
uA0 = d['uv'][t0, jA]
rmap0, rmapn = dist_map(d, t0), dist_map(d, tn)
r_from_map = [float(rmap0[int(round(uA0[1])), int(round(uA0[0]))]),
              float(rmapn[int(round(d['uv'][tn, jA, 1])), int(round(d['uv'][tn, jA, 0]))])]
frames_b = bake_indices(t0, tn, [t0, tn])
Wb, Hb, sb = baked_size(H, W)
Rb = ring_radius(Wb)
print(scene, 't0', t0, 'tn', tn, 'track', jA, 'r', rA[t0], rA[tn], labA, 'from depth map', r_from_map, flush=True)

# distance colormap range over the whole video (same range in stages 0 and 1)
samp = [dist_map(d, i) for i in np.linspace(0, T - 1, 12).astype(int)]
r_lo, r_hi = np.nanpercentile(np.concatenate([s.ravel() for s in samp]), [1, 99])
r_lo, r_hi = max(0.5, float(np.floor(r_lo * 2) / 2)), float(np.ceil(r_hi))
full_idx = np.linspace(0, T - 1, 90).astype(int)  # stages 0-2: the full video as a 9 s time-lapse

# ---------------------------------------------------------------- stage 0
out = []
pw = 410
ph = int(round(H * pw / W))
cb = colorbar(ph + 30, r_lo, r_hi, 'r (m, log)')
for i in full_idx:
    body = blank()
    y = (H_BODY - ph) // 2
    paste(body, fit(d['image'](i), pw), 14, y)
    paste(body, fit(colorize(dist_map(d, i), r_lo, r_hi), pw), 14 + pw + 12, y)
    paste(body, cb, 14 + 2 * pw + 24, y - 24)
    body = text(body, [('RGB', (14, y - 10), 16, (230, 230, 230), True, 'ls'),
                       ('Distance to camera r = z · ‖K⁻¹[u, v, 1]ᵀ‖', (14 + pw + 12, y - 10), 16,
                        (230, 230, 230), True, 'ls'),
                       (f'{scene}   frame {i}/{T - 1}   t = {i / fps:5.1f} s', (14, y + ph + 26), 15, (190, 190, 190), False, 'la'),
                       (f'K: f = {K[0, 0]:.0f} px, c = ({K[0, 2]:.0f}, {K[1, 2]:.0f});  pose: world→camera 4×4 per frame',
                        (14, y + ph + 50), 15, (190, 190, 190), False, 'la')])
    out.append(compose(0, body))
clip0 = save_clip(0, out, len(out) // 2)

# ---------------------------------------------------------------- stage 1
rng = np.random.default_rng(0)
vis_frac = d['vis'].mean(0)
good = np.nonzero(vis_frac > 0.3)[0]
gd, gs = good[dyn[good]], good[~dyn[good]]
pick = np.concatenate([rng.choice(gd, min(25, len(gd)), replace=False), rng.choice(gs, min(50 - min(25, len(gd)), len(gs)), replace=False)])
TR = int(round(fps))  # 1 s trails
out = []
cb = colorbar(300, r_lo, r_hi, 'r (m, log)')
for i in full_idx:
    img = d['image'](i)
    img = cv2.resize(img, (W_OUT - 100, int(round(H * (W_OUT - 100) / W))), interpolation=cv2.INTER_AREA)
    s = img.shape[1] / W
    for j in pick:
        lo = max(0, i - TR)
        seg = d['uv'][lo:i + 1, j] * s
        segv = d['valid'][lo:i + 1, j]
        if segv.sum() > 1:
            trail(img, seg[segv], r_color(d['r'][i, j], r_lo, r_hi) if d['vis'][i, j] else GRAY, 1)
    for j in pick:
        if not d['valid'][i, j]:
            continue
        u, v = d['uv'][i, j] * s
        if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            continue
        c = (int(round(u)), int(round(v)))
        if d['vis'][i, j]:
            cv2.circle(img, c, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, 4, r_color(d['r'][i, j], r_lo, r_hi), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(img, c, 5, GRAY, 2, cv2.LINE_AA)
    body = blank()
    paste(body, img, 0, 0)
    paste(body, cb, W_OUT - 95, 20)
    nv = int(d['vis'][i, pick].sum())
    body = text(body, [(f'{len(pick)} of {d["X"].shape[1]} GT tracks shown', (W_OUT - 95, 340), 12, (220, 220, 220), False, 'la'),
                       (f'visible: {nv}', (W_OUT - 95, 360), 12, (220, 220, 220), False, 'la'),
                       (f'occluded: {len(pick) - nv}', (W_OUT - 95, 378), 12, GRAY, False, 'la'),
                       ('filled = visible', (W_OUT - 95, 404), 12, (220, 220, 220), False, 'la'),
                       ('hollow gray =', (W_OUT - 95, 422), 12, GRAY, False, 'la'),
                       ('  occluded', (W_OUT - 95, 438), 12, GRAY, False, 'la'),
                       ('trail = last 1 s', (W_OUT - 95, 462), 12, (220, 220, 220), False, 'la'),
                       (f't = {i / fps:5.1f} s', (10, img.shape[0] + 22), 15, (190, 190, 190), False, 'la')])
    out.append(compose(1, body))
clip1 = save_clip(1, out, len(out) // 2)

# ---------------------------------------------------------------- stage 2
L = tn - t0
tt = np.arange(T) / fps
vid_w = 480
vid = [fit(d['image'](i), vid_w) for i in range(0, T, max(1, T // 400))]  # small cache for the sweep
vid_i = list(range(0, T, max(1, T // 400)))
nwin = len(win)
per = 2  # windows advanced per output frame
steps = list(range(0, nwin, per)) + [nwin - 1]
chosen = next(k for k, w in enumerate(win) if w['t0'] == t0)
out = []


def stage2_frame(k_done, show_chosen):
    fig, ax = plt.subplots(figsize=(9.4, 1.8), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.2)
    ax.plot(tt, cam, color=(0.2, 0.6, 1.0), lw=1.3, label='camera speed')
    ax.plot(tt, agent, color=(1.0, 0.5, 0.1), lw=1.3, label='max agent speed')
    ax.axhline(CAM_MIN, color=(0.2, 0.6, 1.0), ls=':', lw=1)
    ax.axhline(AGENT_MIN, color=(1.0, 0.5, 0.1), ls=':', lw=1)
    w = win[min(k_done, nwin - 1)]
    ax.axvspan(w['t0'] / fps, w['tn'] / fps, color=(1, 1, 1), alpha=0.15)
    if show_chosen:
        ax.axvspan(t0 / fps, tn / fps, color=np.array(A_COL) / 255, alpha=0.35)
    ax.set_xlim(0, tt[-1])
    ax.set_ylim(0, max(1.0, np.percentile(np.r_[cam, agent], 99) * 1.15))
    ax.set_ylabel('m/s', color='w')
    ax.set_xlabel('time (s)', color='w', labelpad=1)
    ax.legend(loc='upper right', fontsize=8, facecolor=(0.1, 0.1, 0.1), labelcolor='w', framealpha=0.8, ncol=2)
    fig.patch.set_facecolor((18 / 255,) * 3)
    ax.set_facecolor((18 / 255,) * 3)
    ax.tick_params(colors='w', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color((0.5, 0.5, 0.5))
    return fig_to_array(fig)


x0, x1 = int(0.07 * 940), int(0.99 * 940)  # timeline aligned with the plot axes


def tx(t):
    return int(round(x0 + (x1 - x0) * t / tt[-1])) + 10


for k in steps + [nwin - 1] * 25:
    final = len(out) >= len(steps)
    w = win[k]
    body = blank()
    ci = w['t0'] if not final else t0
    fi = vid[min(range(len(vid_i)), key=lambda q: abs(vid_i[q] - ci))]
    paste(body, fi, (W_OUT - vid_w) // 2, 6)
    ytl = 6 + fi.shape[0] + 8
    # timeline: one cell per evaluated window start, green = passes, red = fails
    cv2.rectangle(body, (tx(0), ytl), (tx(tt[-1]), ytl + 16), (60, 60, 60), -1)
    for q in range(k + 1):
        wq = win[q]
        cv2.rectangle(body, (tx(wq['t0'] / fps), ytl), (tx(min(wq['t0'] + int(round(STEP_SEC * fps)), T - 1) / fps), ytl + 16),
                      PASS_COL if wq['passed'] else FAIL_COL, -1)
    if not final:
        cv2.rectangle(body, (tx(w['t0'] / fps), ytl - 4), (tx(w['tn'] / fps), ytl + 20), (255, 255, 255), 2)
    else:
        cv2.rectangle(body, (tx(t0 / fps), ytl - 5), (tx(tn / fps), ytl + 21), A_COL, 3)
    plot = stage2_frame(k, final)
    paste(body, plot, 10, ytl + 26)
    npass = sum(wq['passed'] for wq in win[:k + 1])
    msg = (f'window {k + 1}/{nwin}: t = {w["t0"] / fps:.1f}–{w["tn"] / fps:.1f} s   camera {w["cam"]:.2f} m/s   agent {w["agent"]:.2f} m/s'
           f'   → {"PASS" if w["passed"] else "FAIL"}' if not final else
           f'chosen window: t = {t0 / fps:.2f}–{tn / fps:.2f} s (frames {t0}–{tn})   {npass}/{nwin} windows pass')
    body = text(body, [(msg, (W_OUT // 2, ytl + 26 + plot.shape[0] + 12), 14,
                        A_COL if final else (PASS_COL if w['passed'] else FAIL_COL), True, 'mm'),
                       (f'{N_SEC} s windows, stride {STEP_SEC} s; pass = (camera ≥ {CAM_MIN} or agent ≥ {AGENT_MIN} m/s) '
                        f'and some tracked point changes r by ≥ {M:.0%}', (W_OUT // 2, ytl + 26 + plot.shape[0] + 30), 12,
                        (180, 180, 180), False, 'mm')])
    out.append(compose(2, body))
clip2 = save_clip(2, out, len(out) - 1)

# ---------------------------------------------------------------- stage 3
R_DISP = ring_radius(W)  # ring radius at full (960 px) resolution
out = []
kept_ids = [i for i in reg if reg[i]['keep']]
drop_ids = [i for i in reg if not reg[i]['keep'] and reg[i]['reason'] != 'background']
bg_ids = [i for i in reg if reg[i]['reason'] == 'background']


def region_overlay(i_src, m):
    img = d['image'](i_src).astype(np.float32)
    for i in bg_ids:
        img[m == i] *= 0.55
    for i in drop_ids:
        img[m == i] = img[m == i] * 0.35 + np.array(GRAY) * 0.65
    for i in kept_ids:
        img[m == i] = img[m == i] * 0.45 + np.array(reg[i]['color']) * 0.55
    img = img.astype(np.uint8)
    for i in kept_ids + drop_ids:
        cs, _ = cv2.findContours((m == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, reg[i]['color'] if i in kept_ids else (90, 90, 90), 2, cv2.LINE_AA)
    items = []
    for i in drop_ids:
        ys, xs = np.nonzero(m == i)
        if len(xs):
            items.append((reg[i]['reason'], (int(np.clip(xs.mean(), 70, W - 70)), int(max(ys.min() - 4, 44))), 12,
                          (225, 225, 225), False, 'ms'))
    return img, items


seq3 = [0] * 10 + list(range(tn - t0 + 1))
for q in seq3:
    img, items = region_overlay(t0 + q, wmasks[q])
    cv2.rectangle(img, (0, H - 58), (W, H), (18, 18, 18), -1)
    items += [(f'window frame {q}/{tn - t0}   (t0 + {q / fps:.2f} s)', (10, 16), 14, (230, 230, 230), False, 'la'),
              (f'kept: {len(kept_ids)} object regions', (10, H - 38), 15, (120, 230, 120), True, 'la'),
              (f'dropped: {len(drop_ids)}  (too small < {REG_MIN_AREA:.1%} of image, truncated = touches the border in '
               f'> {REG_MAX_BORDER:.0%} of frames, not visible enough = present in < {REG_MIN_PRESENT:.0%} of frames)',
               (10, H - 16), 12, (200, 200, 200), False, 'la'),
              ('darkened = scene background, not a candidate region', (W - 10, H - 38), 12, (170, 170, 170), False, 'ra')]
    out.append(compose(3, text(img, items)))
clip3 = save_clip(3, out, 10)

# ---------------------------------------------------------------- stage 4
ids0 = track_ids(d, t0, wmasks[0])
pad = marker_pad(d)
rel_all = (d['r'][tn] - d['r'][t0]) / d['r'][t0]
first = d['vis'][t0:t0 + N_MARKED].all(0)
pool = np.nonzero(d['vis'][t0] & np.isin(ids0, kept_ids) & in_bounds(d['uv'][t0], W, H, pad))[0]
rng = np.random.default_rng(1)
rng.shuffle(pool)
fails = {'not visible at tn': [], 'depth variance': [], 'margin': []}
for j in pool:  # first failing check, in pipeline order
    if not (first[j] and d['vis'][tn, j]):
        fails['not visible at tn'].append((int(j), 'occluded at tn'))
        continue
    sd = patch_rel_std(dep0, *d['uv'][t0, j])
    if sd >= STD_MAX:
        if np.isfinite(sd):
            fails['depth variance'].append((int(j), f'\u03c3/\u03bc = {sd:.0%}'))
        continue
    if abs(rel_all[j]) < M:
        fails['margin'].append((int(j), f'\u0394r = {rel_all[j]:+.0%}'))
examples = {k: [] for k in fails}
placed = [uA0]
for why in sorted(fails, key=lambda k: len(fails[k])):  # rarest reason first, so every reason gets tag space
    for j, note in fails[why]:
        if len(examples[why]) >= 2:
            break
        if any(abs(d['uv'][t0, j, 0] - p[0]) < 170 and abs(d['uv'][t0, j, 1] - p[1]) < 50 for p in placed):
            continue
        examples[why].append((j, note))
        placed.append(d['uv'][t0, j])
print('stage 4 rejected examples:', examples, flush=True)
missing4 = [k for k, v in examples.items() if not v]
pd = patch_rel_std(dep0, *uA0)
out = []
for q in [0] * 20 + list(range(tn - t0 + 1)) + [tn - t0] * 5:
    i = t0 + q
    img = d['image'](i)
    items = []
    if q == 0 and len(out) < 20:
        for why, ex in examples.items():
            for j, note in ex:
                u, v = d['uv'][t0, j]
                circle(img, u, v, 9, GRAY)
                cv2.line(img, (int(u) - 6, int(v) - 6), (int(u) + 6, int(v) + 6), GRAY, 2, cv2.LINE_AA)
                items.append((f'✗ {why}', (int(u) + 13, int(v) - 3), 13, (215, 215, 215), True, 'ls'))
                items.append((note, (int(u) + 13, int(v) + 13), 12, (190, 190, 190), False, 'ls'))
        draw_marker(img, *uA0, 'A', R_DISP)
        items.append(('✓ passes all checks', (int(uA0[0]) + R_DISP + 4, int(uA0[1]) + R_DISP + 10), 13, A_COL, True, 'la'))
        items.append((f'visible t0…tn, σ/μ = {pd:.2%}, Δr = {relA:+.0%}',
                      (int(uA0[0]) + R_DISP + 4, int(uA0[1]) + R_DISP + 27), 12, A_COL, False, 'la'))
        items.append(('Candidate points at t0 (inside kept regions)', (10, 16), 15, (240, 240, 240), True, 'la'))
    else:
        trail(img, d['uv'][t0:i + 1, jA], A_COL, 2)
        draw_marker(img, *d['uv'][i, jA], 'A', R_DISP)
        items.append((f'A’s GT track   t0 + {q / fps:.2f} s', (10, 16), 15, (240, 240, 240), True, 'la'))
    cv2.rectangle(img, (0, H - 26), (W, H), (18, 18, 18), -1)
    items.append((f'checks: visible on the first {N_MARKED} frames and at tn;  5×5 depth σ/μ < {STD_MAX:.0%};  '
                  f'|Δr| ≥ {M:.0%} margin;  ring fits inside the frame', (10, H - 8), 12, (200, 200, 200), False, 'ls'))
    out.append(compose(4, text(img, items)))
clip4 = save_clip(4, out, 5)

# ---------------------------------------------------------------- stage 5
tw = np.arange(tn - t0 + 1) / fps
rw = rA[t0:tn + 1]
ylo, yhi = min(rw.min(), rA[t0] * (1 - M)), max(rw.max(), rA[t0] * (1 + M))
ypad = 0.12 * (yhi - ylo)


def stage5_plot(q):
    fig, ax = plt.subplots(figsize=(4.7, 4.3), dpi=100)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.9, bottom=0.13)
    bg = (18 / 255,) * 3
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axhspan(rA[t0] * (1 - M), rA[t0] * (1 + M), color=(0.6, 0.6, 0.6), alpha=0.3, lw=0)
    ax.text(0.03, rA[t0] * (1 + M), f'±{M:.0%} margin', color=(0.8, 0.8, 0.8), fontsize=8, va='bottom')
    ax.axhline(rA[t0], color=(0.8, 0.8, 0.8), ls='--', lw=0.8)
    for x, s in [(0, 't0'), (tw[-1], 'tn')]:
        ax.axvline(x, color='w', ls=':', lw=1)
        ax.text(x, yhi + ypad * 0.9, s, color='w', ha='center', va='top', fontsize=10)
    ax.plot(tw[:q + 1], rw[:q + 1], color=np.array(A_COL) / 255, lw=2.2)
    ax.plot(tw[q], rw[q], 'o', color=np.array(A_COL) / 255, ms=6, mec='k')
    if q == len(tw) - 1:
        ax.plot([0, tw[-1]], [rw[0], rw[-1]], 'o', color=np.array(A_COL) / 255, ms=7, mec='k')
        ax.text(tw[-1] * 0.5, (ylo + yhi) / 2, labA, color=(0.3, 1.0, 0.3), fontsize=26, weight='bold', ha='center',
                va='center', bbox=dict(facecolor=bg, edgecolor=(0.3, 1.0, 0.3), boxstyle='round,pad=0.3'))
    ax.set_xlim(-0.08, tw[-1] + 0.08)
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_xlabel('time since t0 (s)', color='w')
    ax.set_ylabel('r_A (m)', color='w')
    ax.set_title('distance of A to the camera (GT)', color='w', fontsize=10)
    ax.tick_params(colors='w', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color((0.5, 0.5, 0.5))
    return fig_to_array(fig)


out = []
for q in list(range(tn - t0 + 1)) + [tn - t0] * 20:
    i = t0 + q
    img = d['image'](i)
    trail(img, d['uv'][t0:i + 1, jA], A_COL, 2)
    draw_marker(img, *d['uv'][i, jA], 'A', R_DISP)
    u, v = d['uv'][i, jA]
    ty = int(v) - R_DISP - 30 if v > H - 80 else int(v) + R_DISP + 24  # keep the label inside the frame
    img = text(img, [(f'r = {rA[i]:.2f} m', (int(u) + R_DISP + 4, ty), 26, A_COL, True, 'la')])
    left = fit(img, 470)
    body = blank()
    paste(body, left, 10, 60)
    plot = stage5_plot(q)
    paste(body, plot, W_OUT - plot.shape[1] - 6, (H_BODY - plot.shape[0]) // 2)
    done = q == tn - t0
    items = [(f't0 + {q / fps:.2f} s', (10, 50), 15, (230, 230, 230), False, 'ls'),
             (f'r(t0) = {rA[t0]:.2f} m', (10, 60 + left.shape[0] + 28), 17, A_COL, True, 'la')]
    if done:
        items += [(f'r(tn) = {rA[tn]:.2f} m   (Δ = {relA:+.1%})', (10, 60 + left.shape[0] + 54), 17, A_COL, True, 'la'),
                  (f'{"Farther" if relA > 0 else "Closer"}: r(tn) {">" if relA > 0 else "<"} r(t0)·(1 {"+" if relA > 0 else "−"} {M:.2f})',
                   (10, 60 + left.shape[0] + 84), 17, (100, 240, 100), True, 'la')]
    items.append((f'Closer if r(tn) < r(t0)·(1−{M:.2f});  Farther if r(tn) > r(t0)·(1+{M:.2f});  otherwise no item',
                  (10, H_BODY - 14), 13, (190, 190, 190), False, 'la'))
    out.append(compose(5, text(body, items)))
clip5 = save_clip(5, out, len(out) - 1)

# ---------------------------------------------------------------- stage 6
baked = []
for k, i in enumerate(frames_b):  # identical to render.py: INTER_AREA resize, marker on the first N_MARKED frames
    img = cv2.resize(d['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA)
    if k < N_MARKED:
        draw_marker(img, d['uv'][i, jA, 0] * sb, d['uv'][i, jA, 1] * sb, 'A', Rb)
    baked.append(img)
in_fps = round((len(frames_b) - 1) / ((frames_b[-1] - frames_b[0]) / fps), 3)
write_mp4(f'{OUT}/stage6_model_input.mp4', baked, in_fps)  # the exact model input, native size and rate
out = []
for k, img in enumerate(baked):
    disp = cv2.resize(img, (W_OUT, H_BODY), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(disp, (W_OUT - 250, 0), (W_OUT, 54), (18, 18, 18), -1)
    disp = text(disp, [(f'frame {k + 1:2d} / {len(baked)}', (W_OUT - 12, 8), 20, (255, 255, 255), True, 'ra'),
                       (f'source frame {frames_b[k]}   {"marked" if k < N_MARKED else "no marker"}', (W_OUT - 12, 34), 13,
                        A_COL if k < N_MARKED else (190, 190, 190), False, 'ra'),
                       (f'model input: {Wb}×{Hb}, {len(baked)} frames at {in_fps:g} fps (shown 2× enlarged in time, '
                        f'upscaled for display)', (10, H_BODY - 10), 13, (230, 230, 230), False, 'ls')])
    out += [compose(6, disp)] * 2
clip6 = save_clip(6, out, 0)

# ---------------------------------------------------------------- stage 7
question = Q.format(n=N_SEC)
body = blank()
thumb = fit(baked[0], 440)
paste(body, thumb, W_OUT - 440 - 16, 90)
items = [('first input frame (marked)', (W_OUT - 16 - 220, 90 + thumb.shape[0] + 20), 13, (190, 190, 190), False, 'mm'),
         ('Question', (24, 60), 16, (170, 170, 170), True, 'la')]
y = 88
for line in textwrap.wrap(question, 38):
    items.append((line, (24, y), 21, (255, 255, 255), False, 'la'))
    y += 30
y += 20
items.append(('Options', (24, y), 16, (170, 170, 170), True, 'la'))
y += 28
for o in ['Closer', 'Farther']:
    items.append((('● ' if o == labA else '○ ') + o, (40, y), 21, (100, 240, 100) if o == labA else (220, 220, 220), o == labA, 'la'))
    y += 30
y += 20
items.append(('Answer', (24, y), 16, (170, 170, 170), True, 'la'))
items.append((labA, (24, y + 26), 30, (100, 240, 100), True, 'la'))
items.append((f'GT: r(t0) = {rA[t0]:.2f} m → r(tn) = {rA[tn]:.2f} m ({relA:+.0%});  {scene}, frames {t0}–{tn}',
              (24, H_BODY - 24), 13, (170, 170, 170), False, 'la'))
card = compose(7, text(body, items))
clip7 = save_clip(7, [card] * 50, 0)

# ---------------------------------------------------------------- full video, meta, checks
full = []
for k, c in enumerate([clip0, clip1, clip2, clip3, clip4, clip5, clip6, clip7]):
    full += [title_card(k)] * int(1.5 * FPS_OUT) + c
write_mp4(f'{OUT}/walkthrough_full.mp4', full, FPS_OUT)
print(f'walkthrough_full.mp4: {len(full) / FPS_OUT:.1f} s', flush=True)

# consistency checks
proj = d['P_cam'][t0:tn + 1, jA] @ K.T
proj = proj[:, :2] / proj[:, 2:3]
reproj_err = float(np.abs(proj - d['uv'][t0:tn + 1, jA]).max())
dep_err = [float(depth_agree(d['depth'](i), d, i, jA)) for i in range(t0, tn + 1)]
ring_px = []
for k, i in enumerate(frames_b):  # marker present iff the baked frame differs from the plain resize
    plain = cv2.resize(d['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA)
    ring_px.append(int((np.abs(baked[k].astype(int) - plain).sum(-1) > 30).sum()))
checks = dict(reproj_err_px_max=reproj_err, depth_vs_track_rel_err_max=max(dep_err),
              ring_changed_px_per_baked_frame=ring_px,
              ring_only_on_first_3=bool(all(p > 0 for p in ring_px[:N_MARKED]) and not any(ring_px[N_MARKED:])),
              label_consistent=labA == ('Farther' if relA > 0 else 'Closer'), rel_exceeds_margin=bool(abs(relA) >= M))
print(json.dumps(checks), flush=True)

meta = dict(dataset='pointodyssey', video_id=scene, fps=fps, resolution=[W, H], template='T1', n_seconds=N_SEC,
            window=[int(t0), int(tn)], t0=int(t0), tn=int(tn), t0_sec=t0 / fps, tn_sec=tn / fps, track_index=int(jA),
            A_pixel_t0=[float(uA0[0]), float(uA0[1])], A_region_id=int(it['region']),
            r_t0=float(rA[t0]), r_tn=float(rA[tn]), r_from_depth_map=r_from_map, rel_change=float(relA), label=labA,
            question=question, options=['Closer', 'Farther'], answer=labA,
            baked_frames=[int(i) for i in frames_b], baked_resolution=[Wb, Hb], baked_fps=in_fps, ring_radius_px=Rb,
            r_A_window=[round(float(x), 4) for x in rw],
            thresholds=dict(m=M, clarity_min_change=M_SHOW, patch_std=STD_MAX, depth_agree=DEPTH_AGREE, cam_min=CAM_MIN,
                            agent_min=AGENT_MIN, window_stride_s=STEP_SEC, region_min_area=REG_MIN_AREA,
                            region_min_present=REG_MIN_PRESENT, region_max_border=REG_MAX_BORDER, region_bg_area=REG_BG_AREA),
            windows=dict(n=len(win), passed=int(sum(w['passed'] for w in win))),
            regions={str(i): {k: v for k, v in r.items() if k != 'color'} for i, r in reg.items()},
            stage4_rejected={k: [j for j, _ in v] for k, v in examples.items()}, stage4_missing_reasons=missing4,
            distance_colormap_range_m=[r_lo, r_hi], checks=checks)
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
