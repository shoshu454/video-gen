# Stage-by-stage walkthrough clips for T4 (ranking at last frame) and T5 (order flips) on PointOdyssey.
# Same layout as walkthrough_pointodyssey.py (T1): one video, one window, the query points across stages 0-7;
# every number comes from PointOdyssey GT (depth, camera, 3D tracks, visibility, instance masks).
# Writes to <OUT_ROOT>/walkthrough_<T>/: stage{k}_<name>.mp4, stage{k}.png, walkthrough_full.mp4, meta.json
#   python walkthrough_po_rank_flip.py --template T4 --select     # scan test scenes, write selection.json
#   python walkthrough_po_rank_flip.py --template T4 [--pick k]   # render the k-th ranked item
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

ap = argparse.ArgumentParser()
ap.add_argument('--template', choices=['T4', 'T5'], required=True)
ap.add_argument('--select', action='store_true')
ap.add_argument('--scenes', default='', help='comma list; default all test scenes')
ap.add_argument('--pick', type=int, default=0)
args = ap.parse_args()
TPL = args.template
OUT = f'{OUT_ROOT}/walkthrough_{TPL}'
os.makedirs(OUT, exist_ok=True)

FPS_OUT = 10
W_OUT, H_BAN, H_BODY = 960, 64, 540
N_SEC = 3 if TPL == 'T4' else 6  # T4: spec gives no length, 3 s; T5: spec says 5-8 s
LETTERS = ['A', 'B', 'C'] if TPL == 'T4' else ['A', 'B']
M = 0.10  # relative margin |r_i - r_j| / min(r_i, r_j), synthetic data
M_SHOW = 0.20 if TPL == 'T4' else 0.15  # walkthrough item: larger margins, for visual clarity
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
N_SAMP = 300  # tracks sampled per window for the stage-2 predicate and the search

GRAY = (150, 150, 150)
PASS_COL, FAIL_COL = (44, 160, 44), (214, 39, 40)
REGION_COLS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]  # no yellow/cyan/magenta
FONT_DIR = f'{matplotlib.get_data_path()}/fonts/ttf'
STAGES = [(0, 'standardize', 'Standardize', 'Common format: RGB, intrinsics, poses, per-pixel distance to camera.'),
          (1, 'geometry', 'Geometry', '3D point tracks + visibility for every frame.'),
          (2, 'clips', 'Clips', 'Slide windows; keep those with enough motion.'),
          (3, 'regions', 'Regions', 'Pick valid regions from instance masks.'),
          (4, 'points', 'Points', 'Place the query points; keep them only if they pass the checks.'),
          (5, 'labels', 'Labels', 'Compute the answer from GT distance.'),
          (6, 'render', 'Render', 'Bake frames; mark the points on the first 3 frames only.'),
          (7, 'question', 'Question', 'Final QA item.')]
Q = {'T4': 'Rank A, B and C from closest to farthest from the camera in the last frame.',
     'T5': 'Throughout the video, how many times does the closer of A and B switch?'}[TPL]


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


def rel_margin(ri, rj):
    return np.abs(ri - rj) / np.minimum(ri, rj)


def flip_trace(ra, rb):
    """T5 label rule on baked frames. ra, rb: (K, P) distances of A and B.
    state: +1 = A closer, -1 = B closer, 0 = margin < M (no order). The order at t0 must hold with margin >= M on
    baked frames 0 and 1 (init_ok). A flip is counted at frame k when the opposite order holds with margin >= M on
    frames k and k+1 (>= 2 consecutive baked frames). Returns state, flip_at (K, P) bool, init_ok (P,), signed margin."""
    sm = (rb - ra) / np.minimum(ra, rb)  # > 0: A closer
    state = np.where(np.abs(sm) >= M, np.sign(sm), 0).astype(np.int8)
    init_ok = (state[0] != 0) & (state[1] == state[0])
    cur = state[0].copy()
    flip_at = np.zeros(state.shape, bool)
    for k in range(1, len(state) - 1):
        f = init_ok & (state[k] == -cur) & (state[k + 1] == -cur)
        flip_at[k] = f
        cur = np.where(f, -cur, cur)
    return state, flip_at, init_ok, sm


def flip_label(n):
    return 'Never' if n == 0 else 'Once' if n == 1 else 'More than once'


def window_bake(t0, tn):
    return np.array(bake_indices(t0, tn, [t0, tn]))


def windows(d, cam, agent):
    """Motion predicate + template predicate on N_SAMP sampled tracks.
    T4: some pair visible at t0 and tn keeps margin >= M at both ends and swaps order (the ranking can change).
    T5: some pair visible on the baked frames flips at least once under the T5 rule."""
    L = int(round(N_SEC * d['fps']))
    step = int(round(STEP_SEC * d['fps']))
    rng = np.random.default_rng(0)
    out = []
    for t0 in range(0, d['n_frames'] - L, step):
        tn = t0 + L
        fb = window_bake(t0, tn)
        c, g = float(cam[t0:tn + 1].mean()), float(agent[t0:tn + 1].mean())
        motion = bool(c >= CAM_MIN or g >= AGENT_MIN)
        if TPL == 'T4':
            ok = np.nonzero(d['vis'][fb[:N_MARKED]].all(0) & d['vis'][tn])[0]
            s = rng.permutation(ok)[:N_SAMP]
            i, j = np.triu_indices(len(s), 1)
            r0, rn = d['r'][t0, s], d['r'][tn, s]
            tp = bool(((rel_margin(r0[i], r0[j]) >= M) & (rel_margin(rn[i], rn[j]) >= M)
                       & (np.sign(r0[i] - r0[j]) != np.sign(rn[i] - rn[j]))).any())
        else:
            ok = np.nonzero(d['vis'][fb].all(0))[0]
            s = rng.permutation(ok)[:N_SAMP]
            i, j = np.triu_indices(len(s), 1)
            R = d['r'][fb][:, s]
            _, flip_at, _, _ = flip_trace(R[:, i], R[:, j])
            tp = bool(flip_at.any())
        out.append(dict(t0=t0, tn=tn, cam=c, agent=g, motion=motion, tpl=tp, passed=motion and tp))
    return out


def regions(d, t0, tn):
    """Instance regions of the window. Returns {id: dict(keep, reason, color)} and the masks."""
    masks = [d['mask'](i) for i in range(t0, tn + 1)]
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


def min_spacing(uv):
    """uv: (F, P, k, 2) marker positions -> (P,) smallest pairwise pixel distance over the F frames."""
    k = uv.shape[2]
    dm = [np.linalg.norm(uv[:, :, a] - uv[:, :, b], axis=-1).min(0) for a in range(k) for b in range(a + 1, k)]
    return np.min(dm, 0)


# ---------------------------------------------------------------- selection
if args.select:
    scenes = args.scenes.split(',') if args.scenes else [s for s in po_scenes() if s.startswith('test/')]
    cands = []
    for sc in scenes:
        d = load(sc)
        W, H, fps = d['W'], d['H'], d['fps']
        cam, agent, dyn = motion_summary(d)
        win = windows(d, cam, agent)
        pad = marker_pad(d)
        print(sc, d['n_frames'], 'cam p50/p90 %.2f/%.2f' % tuple(np.percentile(cam, [50, 90])),
              'agent p50/p90 %.2f/%.2f' % tuple(np.percentile(agent, [50, 90])),
              'windows pass %d/%d' % (sum(w['passed'] for w in win), len(win)), flush=True)
        rng = np.random.default_rng(0)
        sc_c = []
        cnt = dict(windows=0, cand_vis=0, cand_depth=0, cand_region=0, combos_ok=0, stored=0, region_fail=0)
        for w in win:
            t0, tn = w['t0'], w['tn']
            if not w['passed'] or t0 % int(round(fps)):  # search every 1 s
                continue
            fb = window_bake(t0, tn)
            m0 = d['mask'](t0)
            ids = track_ids(d, t0, m0)
            area = np.bincount(m0.ravel()) / m0.size
            reg_ok = np.zeros(len(ids), bool)
            reg_ok[ids >= 0] = (area[ids[ids >= 0]] >= REG_MIN_AREA) & (area[ids[ids >= 0]] <= REG_BG_AREA)
            # clarity: every query point visible (GT flag) and in frame on every window frame
            okv = d['vis'][t0:tn + 1].all(0) & in_bounds(d['uv'][t0:tn + 1], W, H, 20).all(0) \
                & in_bounds(d['uv'][fb[:N_MARKED]], W, H, pad).all(0) & reg_ok
            cand = rng.permutation(np.nonzero(okv)[0])[:N_SAMP]
            cnt['windows'] += 1
            cnt['cand_vis'] += len(cand)
            # per-point clarity checks before the search: 5x5 depth sigma/mu at t0, and the depth map agrees with the
            # track on every frame that is shown (all window frames for T4, the baked frames for T5)
            dep0 = d['depth'](t0)
            cand = cand[[patch_rel_std(dep0, *d['uv'][t0, j]) < STD_MAX for j in cand]]
            for i in (range(t0, tn + 1) if TPL == 'T4' else fb):
                if len(cand) < len(LETTERS):
                    break
                dep = dep0 if i == t0 else d['depth'](i)
                u, v = np.round(d['uv'][i, cand]).astype(int).T
                z = d['P_cam'][i, cand, 2]
                cand = cand[np.abs(dep[v, u] - z) / z <= DEPTH_AGREE]
            cnt['cand_depth'] += len(cand)
            if len(cand) < len(LETTERS):
                continue
            reg, masks = regions(d, t0, tn)  # full stage-3 rules; points must sit in kept regions
            del masks
            cand = cand[[ids[j] in reg and reg[ids[j]]['keep'] for j in cand]]
            cnt['cand_region'] += len(cand)
            if len(cand) < len(LETTERS):
                continue
            if TPL == 'T4':
                J = rng.integers(0, len(cand), (20000, 3))
                J = cand[J[(J[:, 0] != J[:, 1]) & (J[:, 0] != J[:, 2]) & (J[:, 1] != J[:, 2])]]
                r0, rn = d['r'][t0][J], d['r'][tn][J]
                pr = [(0, 1), (0, 2), (1, 2)]
                mn = np.min([rel_margin(rn[:, a], rn[:, b]) for a, b in pr], 0)
                m0v = np.min([rel_margin(r0[:, a], r0[:, b]) for a, b in pr], 0)
                changed = (np.argsort(r0, 1) != np.argsort(rn, 1)).any(1)
                sp = min_spacing(d['uv'][fb[:N_MARKED]][:, J])
                rid = ids[J]
                ndist = 1 + (rid[:, 0] != rid[:, 1]) + ((rid[:, 2] != rid[:, 0]) & (rid[:, 2] != rid[:, 1]))
                ok = (mn >= M_SHOW) & (m0v >= M) & changed & (sp >= pad) & (ndist >= 2)
                score = np.minimum(mn, 0.5) + 0.5 * np.minimum(m0v, 0.3) + 0.3 * (ndist == 3) + 0.1 * dyn[J].sum(1) \
                    + 0.2 * (w['cam'] >= CAM_MIN) + 0.2 * (w['agent'] >= AGENT_MIN)
                cnt['combos_ok'] += int(ok.sum())
                for q in np.argsort(-np.where(ok, score, -1))[:3]:
                    if ok[q]:
                        sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=[int(x) for x in J[q]], score=float(score[q]),
                                         min_margin_tn=float(mn[q]), min_margin_t0=float(m0v[q]), cam=w['cam'],
                                         agent=w['agent']))
            else:
                i, j = np.triu_indices(len(cand), 1)
                R = d['r'][fb][:, cand]
                state, flip_at, init_ok, sm = flip_trace(R[:, i], R[:, j])
                nf = flip_at.sum(0)
                JA, JB = cand[i], cand[j]
                sp = min_spacing(d['uv'][fb[:N_MARKED]][:, np.stack([JA, JB], 1)])
                ok = init_ok & (nf >= 1) & (sp >= pad) & (ids[JA] != ids[JB])
                clear = (state != 0).mean(0)
                # each order run should be clear: peak |margin| between consecutive flips
                peak = np.full(len(JA), np.inf)
                seg = np.cumsum(flip_at, 0)
                for s_ in range(int(nf.max()) + 1 if len(nf) else 0):
                    pk = np.where(seg == s_, np.abs(sm), 0).max(0)
                    peak = np.where(s_ <= nf, np.minimum(peak, pk), peak)
                ok &= (peak >= M_SHOW) & (clear >= 0.7)
                score = np.minimum(peak, 0.5) + 0.5 * clear + 0.3 * (nf == 1) + 0.4 * (nf == 2) \
                    + 0.1 * (dyn[JA].astype(int) + dyn[JB]) + 0.2 * (w['cam'] >= CAM_MIN) + 0.2 * (w['agent'] >= AGENT_MIN)
                cnt['combos_ok'] += int(ok.sum())
                n_ok = 0
                for q in np.argsort(-np.where(ok, score, -1)):
                    if not ok[q] or n_ok >= 3:
                        break
                    # clarity: every order run (before, between and after flips) lasts >= 4 baked frames
                    fk = np.nonzero(flip_at[:, q])[0]
                    if np.diff(np.r_[0, fk, len(fb)]).min() < 4:
                        continue
                    n_ok += 1
                    if True:
                        sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=[int(JA[q]), int(JB[q])], score=float(score[q]),
                                         n_flips=int(nf[q]), label=flip_label(int(nf[q])), peak_margin=float(peak[q]),
                                         clear_frac=float(clear[q]), cam=w['cam'], agent=w['agent']))
        # verify the best few per scene: every point inside a kept region (full stage-3 rules), >= 2 distinct regions
        sc_c.sort(key=lambda c: -c['score'])
        cnt['stored'] = len(sc_c)
        used, kept, reg_cache = set(), 0, {}
        for c in sc_c:
            if kept >= 3 or len(reg_cache) >= 12:
                break
            t0, tn = c['t0'], c['tn']
            if t0 in used:
                continue
            if t0 not in reg_cache:
                reg, masks = regions(d, t0, tn)
                reg_cache[t0] = (reg, track_ids(d, t0, masks[0]))
                del masks
            reg, ids0_ = reg_cache[t0]
            rid = [int(x) for x in ids0_[c['j']]]
            if not all(x in reg and reg[x]['keep'] for x in rid) or len(set(rid)) < 2:
                cnt['region_fail'] += 1
                continue
            used.add(t0)
            c['regions'] = rid
            cands.append(c)
            kept += 1
            print('  cand', json.dumps(c), flush=True)
        print('  counts', json.dumps(cnt), flush=True)
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
    b[-3:] = COLORS['A']
    return text(b, [(f'Stage {k} — {name}', (14, 8), 24, (255, 255, 255), True, 'la'),
                    (line, (14, 40), 16, (210, 210, 210), False, 'la'),
                    (f'{TPL} — {"ranking at last frame" if TPL == "T4" else "order flips"}', (W_OUT - 14, 10), 16,
                     (200, 200, 200), True, 'ra')])


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
    write_mp4(f'{OUT}/{name}.mp4', frames, FPS_OUT)
    Image.fromarray(frames[still]).save(f'{OUT}/stage{k}.png')
    print(f'{name}.mp4: {len(frames)} frames = {len(frames) / FPS_OUT:.1f} s, still = frame {still}', flush=True)
    return frames


def r_labels(img, i, R, size=20):
    """Draw each point's marker and live distance at source frame i; labels are kept inside the frame."""
    items = []
    for L, j in P.items():
        u, v = d['uv'][i, j]
        draw_marker(img, u, v, L, R)
        tx_ = int(u) + R + 4 if u < W - 150 else int(u) - R - 4
        ty = int(v) + R + 22 if v < H - 50 else int(v) - R - 8
        items.append((f'{d["r"][i, j]:.2f} m', (tx_, ty), size, COLORS[L], True, 'ls' if u < W - 150 else 'rs'))
    return items


# ---------------------------------------------------------------- the item
sel = json.load(open(f'{OUT}/selection.json'))
it = sel[args.pick]
scene, t0, tn = it['scene'], it['t0'], it['tn']
P = dict(zip(LETTERS, it['j']))  # letter -> track index
d = load(scene)
T, fps, H, W, K = d['n_frames'], d['fps'], d['H'], d['W'], d['K']
cam, agent, dyn = motion_summary(d)
win = windows(d, cam, agent)
reg, wmasks = regions(d, t0, tn)
frames_b = window_bake(t0, tn)
Wb, Hb, sb = baked_size(H, W)
Rb = ring_radius(Wb)
R_DISP = ring_radius(W)  # ring radius at full (960 px) resolution
WSTEP = 1 if TPL == 'T4' else 3  # stages 3-4 play every WSTEP-th window frame
r_pt = {L: d['r'][:, j] for L, j in P.items()}
if TPL == 'T4':
    rank0 = sorted(P, key=lambda L: r_pt[L][t0])
    rankn = sorted(P, key=lambda L: r_pt[L][tn])
    label = ', '.join(rankn)
    # options: the correct permutation + the three one-swap permutations, shuffled; answer = option letter
    perms = [rankn] + [[rankn[b] if x == a else rankn[a] if x == b else rankn[x] for x in range(3)]
                       for a, b in [(0, 1), (1, 2), (0, 2)]]
    order = np.random.default_rng(0).permutation(4)
    OPT_LET = ['a', 'b', 'c', 'd']  # lowercase so option letters are not confused with the point letters
    options = [f'{OPT_LET[k]}) ' + ', '.join(perms[o]) for k, o in enumerate(order)]
    answer = OPT_LET[int(np.nonzero(order == 0)[0][0])]
    margins_tn = {f'{a}{b}': float(rel_margin(r_pt[a][tn], r_pt[b][tn])) for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]}
    margins_t0 = {f'{a}{b}': float(rel_margin(r_pt[a][t0], r_pt[b][t0])) for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]}
else:
    RB = np.stack([r_pt['A'][frames_b], r_pt['B'][frames_b]], 1)
    state, flip_at, init_ok, sm = flip_trace(RB[:, :1], RB[:, 1:])
    state, flip_at, sm = state[:, 0], flip_at[:, 0], sm[:, 0]
    assert init_ok[0]
    flips_k = [int(k) for k in np.nonzero(flip_at)[0]]
    label = flip_label(len(flips_k))
    options = ['Never', 'Once', 'More than once']
    answer = label
print(scene, 't0', t0, 'tn', tn, 'points', P, 'label', label, flush=True)

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
tt = np.arange(T) / fps
vid_w = 480
vid = [fit(d['image'](i), vid_w) for i in range(0, T, max(1, T // 400))]  # small cache for the sweep
vid_i = list(range(0, T, max(1, T // 400)))
nwin = len(win)
per = 2  # windows advanced per output frame
steps = list(range(0, nwin, per)) + [nwin - 1]
out = []
A_COL = COLORS['A']
TPL_RULE = {'T4': f'some tracked pair keeps a ≥ {M:.0%} margin at t0 and at the last frame and swaps order',
            'T5': f'some tracked pair flips order at least once (≥ {M:.0%} margin, held ≥ 2 baked frames)'}[TPL]


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
    dark_axes(fig, [ax])
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
    why = '' if w['passed'] else ('  (too little motion)' if not w['motion'] else f'  (no {"swap" if TPL == "T4" else "flip"})')
    msg = (f'window {k + 1}/{nwin}: t = {w["t0"] / fps:.1f}–{w["tn"] / fps:.1f} s   camera {w["cam"]:.2f} m/s   agent {w["agent"]:.2f} m/s'
           f'   → {"PASS" if w["passed"] else "FAIL"}{why}' if not final else
           f'chosen window: t = {t0 / fps:.2f}–{tn / fps:.2f} s (frames {t0}–{tn})   {npass}/{nwin} windows pass')
    body = text(body, [(msg, (W_OUT // 2, ytl + 26 + plot.shape[0] + 12), 14,
                        A_COL if final else (PASS_COL if w['passed'] else FAIL_COL), True, 'mm'),
                       (f'{N_SEC} s windows, stride {STEP_SEC} s; pass = (camera ≥ {CAM_MIN} or agent ≥ {AGENT_MIN} m/s) '
                        f'and {TPL_RULE}', (W_OUT // 2, ytl + 26 + plot.shape[0] + 30), 11,
                        (180, 180, 180), False, 'mm')])
    out.append(compose(2, body))
clip2 = save_clip(2, out, len(out) - 1)

# ---------------------------------------------------------------- stage 3
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


seq3 = [0] * 10 + list(range(0, tn - t0 + 1, WSTEP))
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
# Rejected examples: other candidates tried in place of the last point (C for T4, B for T5), first failing check.
ids0 = track_ids(d, t0, wmasks[0])
pad = marker_pad(d)
dep0 = d['depth'](t0)
first = d['vis'][frames_b[:N_MARKED]].all(0)
fixed = LETTERS[:-1]
pool = np.nonzero(d['vis'][t0] & np.isin(ids0, kept_ids) & in_bounds(d['uv'][t0], W, H, pad))[0]
pool = pool[~np.isin(pool, list(P.values()))]
rng = np.random.default_rng(1)
rng.shuffle(pool)
pool = pool[:3000]
if TPL == 'T4':
    fails = {'not visible at t0 / last frame': [], 'depth variance': [], 'too close to a marker': [], 'margin at last frame': []}
else:
    fails = {'not visible at t0': [], 'depth variance': [], 'too close to A': [], 'order unclear at t0': [],
             'occluded at a flip': []}
    Rp = d['r'][frames_b][:, pool]
    st_p, fa_p, init_p, _ = flip_trace(np.repeat(RB[:, :1], len(pool), 1), Rp)
for q_, j in enumerate(pool):  # first failing check, in pipeline order
    near = min(np.linalg.norm(d['uv'][frames_b[:N_MARKED], j] - d['uv'][frames_b[:N_MARKED], P[L]], axis=-1).min() for L in fixed)
    if TPL == 'T4':
        if not (first[j] and d['vis'][tn, j]):
            fails['not visible at t0 / last frame'].append((int(j), 'occluded at the last frame' if first[j] else 'occluded at t0'))
            continue
    elif not first[j]:
        fails['not visible at t0'].append((int(j), 'occluded on a marked frame'))
        continue
    sd = patch_rel_std(dep0, *d['uv'][t0, j])
    if sd >= STD_MAX:
        if np.isfinite(sd):
            fails['depth variance'].append((int(j), f'σ/μ = {sd:.1%}'))
        continue
    if near < pad:
        fails['too close to a marker' if TPL == 'T4' else 'too close to A'].append((int(j), f'{near:.0f} px < {pad:.0f} px'))
        continue
    if TPL == 'T4':
        mm = min(rel_margin(d['r'][tn, j], r_pt[L][tn]) for L in fixed)
        if mm < M:
            fails['margin at last frame'].append((int(j), f'min margin = {mm:.1%}'))
    else:
        if not init_p[q_]:
            fails['order unclear at t0'].append((int(j), f'|margin| = {rel_margin(Rp[0, q_], RB[0, 0]):.1%}, '
                                                          f'{rel_margin(Rp[1, q_], RB[1, 0]):.1%} on frames 1–2'))
            continue
        fk = np.nonzero(fa_p[:, q_])[0]
        occ = [k for k in fk if not (d['vis'][frames_b[k - 1:k + 1], j].all() and d['vis'][frames_b[k - 1:k + 1], P['A']].all())]
        if occ:
            fails['occluded at a flip'].append((int(j), f'flip at baked frame {occ[0] + 1}'))
LEG_W = 470
LEG_H = 22 * (len(LETTERS) + 3) + 12
# bottom-left panel unless a chosen point sits there at t0, then bottom-right
LEG_X = 0 if not any(d['uv'][t0, j, 0] < LEG_W + 20 and d['uv'][t0, j, 1] > H - 26 - LEG_H - 20 for j in P.values()) \
    else W - LEG_W
examples = {k: [] for k in fails}
placed = [d['uv'][t0, j] for j in P.values()]
for why in sorted(fails, key=lambda k: len(fails[k])):  # rarest reason first, so every reason gets tag space
    for j, note in fails[why]:
        if len(examples[why]) >= (2 if TPL == 'T5' else 2):
            break
        if any(abs(d['uv'][t0, j, 0] - p[0]) < 170 and abs(d['uv'][t0, j, 1] - p[1]) < 50 for p in placed):
            continue
        if not (60 < d['uv'][t0, j, 1] < H - 90 and d['uv'][t0, j, 0] < W - 170):
            continue
        if LEG_X - 30 < d['uv'][t0, j, 0] < LEG_X + LEG_W and d['uv'][t0, j, 1] > H - 26 - LEG_H - 30:  # legend panel
            continue
        examples[why].append((j, note))
        placed.append(d['uv'][t0, j])
print('stage 4 rejected examples:', examples, flush=True)
missing4 = [k for k, v in examples.items() if not v]
pstat = {L: dict(sd=patch_rel_std(dep0, *d['uv'][t0, j]), rid=int(ids0[j])) for L, j in P.items()}
spacing = float(min(np.linalg.norm(d['uv'][frames_b[:N_MARKED], P[a]] - d['uv'][frames_b[:N_MARKED], P[b]], axis=-1).min()
                    for ai, a in enumerate(LETTERS) for b in LETTERS[ai + 1:]))


def legend_box(img, lines):
    """Bottom panel listing each point's check values."""
    y0 = H - 26 - LEG_H
    sub = img[y0:H - 26, LEG_X:LEG_X + LEG_W].astype(np.float32)
    img[y0:H - 26, LEG_X:LEG_X + LEG_W] = (sub * 0.35 + 18 * 0.65).astype(np.uint8)
    return [(s, (LEG_X + 10, y0 + 8 + 22 * n), 14, col, bold, 'la') for n, (s, col, bold) in enumerate(lines)]


out = []
for q in [0] * 20 + list(range(0, tn - t0 + 1, WSTEP)) + [tn - t0] * 5:
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
        lines = [('✓ chosen points pass all checks', (120, 230, 120), True)]
        for L, j in P.items():
            extra = f'r(last) = {r_pt[L][tn]:.2f} m' if TPL == 'T4' else f'r(t0) = {r_pt[L][t0]:.2f} m'
            lines.append((f'{L}: region {pstat[L]["rid"]}, σ/μ = {pstat[L]["sd"]:.2%}, {extra}', COLORS[L], False))
        if TPL == 'T4':
            lines.append((f'last-frame margins: ' + ', '.join(f'{k} {v:.0%}' for k, v in margins_tn.items()), (230, 230, 230), False))
        else:
            lines.append((f'margin at t0: {abs(sm[0]):.0%}  ({"A" if sm[0] > 0 else "B"} closer)', (230, 230, 230), False))
        lines.append((f'min marker spacing {spacing:.0f} px ≥ {pad:.0f} px', (230, 230, 230), False))
        items += legend_box(img, lines)
        for L, j in P.items():  # after the panel, so the panel never dims a marker
            draw_marker(img, *d['uv'][t0, j], L, R_DISP)
        items.append(('Candidate points at t0 (inside kept regions)', (10, 16), 15, (240, 240, 240), True, 'la'))
    else:
        for L, j in P.items():
            trail(img, d['uv'][t0:i + 1, j], COLORS[L], 2)
        for L, j in P.items():
            draw_marker(img, *d['uv'][i, j], L, R_DISP)
        items.append((f'GT tracks   t0 + {q / fps:.2f} s', (10, 16), 15, (240, 240, 240), True, 'la'))
    cv2.rectangle(img, (0, H - 26), (W, H), (18, 18, 18), -1)
    rule = {'T4': f'checks: visible on the first {N_MARKED} frames and the last frame;  5×5 depth σ/μ < {STD_MAX:.0%};  '
                  f'markers ≥ 3 ring radii apart;  pairwise margin ≥ {M:.0%} at the last frame;  ≥ 2 regions',
            'T5': f'checks: visible on the first {N_MARKED} frames and around each flip;  5×5 depth σ/μ < {STD_MAX:.0%};  '
                  f'markers ≥ 3 ring radii apart;  order clear at t0 (margin ≥ {M:.0%})'}[TPL]
    items.append((rule, (10, H - 8), 11, (200, 200, 200), False, 'ls'))
    out.append(compose(4, text(img, items)))
clip4 = save_clip(4, out, 5)

# ---------------------------------------------------------------- stage 5
out = []
if TPL == 'T4':
    tw = np.arange(tn - t0 + 1) / fps
    rw = {L: r_pt[L][t0:tn + 1] for L in P}
    allr = np.concatenate(list(rw.values()))
    ylo, yhi = allr.min(), allr.max()
    ypad = 0.15 * (yhi - ylo)

    def stage5_plot(q):
        fig, ax = plt.subplots(figsize=(4.7, 4.3), dpi=100)
        fig.subplots_adjust(left=0.16, right=0.88, top=0.9, bottom=0.13)
        bg = dark_axes(fig, [ax])
        for x, s in [(0, 't0'), (tw[-1], 'last')]:
            ax.axvline(x, color='w', ls=':', lw=1)
            ax.text(x, yhi + ypad * 0.9, s, color='w', ha='center', va='top', fontsize=10)
        for L in P:
            ax.plot(tw[:q + 1], rw[L][:q + 1], color=mcol(L), lw=2.2)
            ax.plot(tw[q], rw[L][q], 'o', color=mcol(L), ms=6, mec='k')
        for n, L in enumerate(rank0):  # ranking at t0 (gray, left)
            ax.text(-0.05, rw[L][0], f'{n + 1}', color=(0.75, 0.75, 0.75), fontsize=9, ha='right', va='center')
        if q == len(tw) - 1:
            for n, L in enumerate(rankn):
                ax.text(tw[-1] + 0.06, rw[L][-1], f'{n + 1}. {L}', color=mcol(L), fontsize=11, weight='bold', va='center')
        ax.set_xlim(-0.15, tw[-1] + 0.08)
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_xlabel('time since t0 (s)', color='w')
        ax.set_ylabel('r (m)', color='w')
        ax.set_title('distance of each point to the camera (GT)', color='w', fontsize=10)
        return fig_to_array(fig)

    for q in list(range(tn - t0 + 1)) + [tn - t0] * 25:
        i = t0 + q
        img = d['image'](i)
        for L, j in P.items():
            trail(img, d['uv'][t0:i + 1, j], COLORS[L], 2)
        img = text(img, r_labels(img, i, R_DISP, 24))
        left = fit(img, 470)
        body = blank()
        paste(body, left, 10, 60)
        plot = stage5_plot(q)
        paste(body, plot, W_OUT - plot.shape[1] - 6, (H_BODY - plot.shape[0]) // 2)
        cur = sorted(P, key=lambda L: r_pt[L][i])
        items = [(f't0 + {q / fps:.2f} s', (10, 50), 15, (230, 230, 230), False, 'ls'),
                 ('ranking at t0:   ' + ', '.join(rank0), (10, 60 + left.shape[0] + 28), 17, (200, 200, 200), True, 'la'),
                 ('current ranking: ' + ', '.join(cur), (10, 60 + left.shape[0] + 54), 17, (230, 230, 230), False, 'la')]
        if q == tn - t0:
            items += [(f'last frame: {label}   (closest → farthest)', (10, 60 + left.shape[0] + 84), 19, (100, 240, 100), True, 'la'),
                      ('margins: ' + ', '.join(f'{k} {v:.0%}' for k, v in margins_tn.items()) + f'  (all ≥ {M:.0%})',
                       (10, 60 + left.shape[0] + 112), 14, (200, 200, 200), False, 'la')]
        items.append((f'label = order of r at the last frame; every pair must differ by ≥ {M:.0%}, otherwise no item',
                      (10, H_BODY - 12), 13, (190, 190, 190), False, 'ls'))
        out.append(compose(5, text(body, items)))
else:
    tw = np.arange(tn - t0 + 1) / fps
    tb = (frames_b - t0) / fps
    ST_COL = {1: mcol('A'), -1: mcol('B'), 0: (0.5, 0.5, 0.5)}
    rw = {L: r_pt[L][t0:tn + 1] for L in P}
    allr = np.concatenate(list(rw.values()))
    lim = max(0.3, float(np.abs(sm).max()) * 1.15)

    def stage5_plot(k):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.7, 4.5), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.11, hspace=0.12)
        bg = dark_axes(fig, [ax, bx])
        q = frames_b[k] - t0
        for L in P:
            ax.plot(tw[:q + 1], rw[L][:q + 1], color=mcol(L), lw=2, label=L)
            ax.plot(tb[:k + 1], r_pt[L][frames_b[:k + 1]], 'o', color=mcol(L), ms=3, mec='k', mew=0.5)
        ax.set_xlim(-0.1, tw[-1] + 0.1)
        ax.set_ylim(allr.min() - 0.1 * np.ptp(allr), allr.max() + 0.1 * np.ptp(allr))
        ax.set_ylabel('r (m)', color='w')
        ax.set_title('GT distance; dots = baked frames', color='w', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, facecolor=(0.1, 0.1, 0.1), labelcolor='w', framealpha=0.8, ncol=2)
        bx.axhspan(-M, M, color=(0.6, 0.6, 0.6), alpha=0.3, lw=0)
        bx.axhline(0, color=(0.7, 0.7, 0.7), lw=0.6)
        bx.text(tw[-1], M, f'±{M:.0%}: no order', color=(0.8, 0.8, 0.8), fontsize=7, va='bottom', ha='right')
        bx.bar(tb[:k + 1], sm[:k + 1], width=0.8 * (tb[1] - tb[0]), color=[ST_COL[int(s_)] for s_ in state[:k + 1]])
        n = 0
        for f in flips_k:
            if f <= k:
                n += 1
                bx.axvline(tb[f] - 0.5 * (tb[f] - tb[f - 1]), color=(0.3, 1.0, 0.3), ls='--', lw=1.2)
                xf = tb[f] - 0.5 * (tb[f] - tb[f - 1])
                bx.text(xf + (0.08 if xf < tw[-1] * 0.7 else -0.08), -lim * 0.6 * state[f], f'flip {n}',
                        color=(0.3, 1.0, 0.3), fontsize=9, ha='left' if xf < tw[-1] * 0.7 else 'right', va='center', weight='bold')
        bx.set_ylim(-lim, lim)
        bx.set_ylabel('(r_B − r_A) / min', color='w', fontsize=8)
        bx.set_xlabel('time since t0 (s)', color='w')
        bx.text(0.02, lim * 0.9, 'A closer ↑', color=mcol('A'), fontsize=8, ha='left', va='top')
        bx.text(0.02, -lim * 0.9, 'B closer ↓', color=mcol('B'), fontsize=8, ha='left', va='bottom')
        return fig_to_array(fig)

    for k in [x for k in range(len(frames_b)) for x in [k] * 3] + [len(frames_b) - 1] * 22:
        i = frames_b[k]
        img = d['image'](i)
        for L, j in P.items():
            trail(img, d['uv'][t0:i + 1, j], COLORS[L], 2)
        img = text(img, r_labels(img, i, R_DISP, 24))
        left = fit(img, 470)
        body = blank()
        paste(body, left, 10, 60)
        plot = stage5_plot(k)
        paste(body, plot, W_OUT - plot.shape[1] - 6, (H_BODY - plot.shape[0]) // 2)
        nf = sum(f <= k for f in flips_k)
        st_txt = {1: f'A closer (margin {abs(sm[k]):.0%})', -1: f'B closer (margin {abs(sm[k]):.0%})',
                  0: f'within ±{M:.0%}: no order'}[int(state[k])]
        items = [(f'baked frame {k + 1}/{len(frames_b)}   (source {i}, t0 + {(i - t0) / fps:.2f} s)', (10, 50), 15, (230, 230, 230), False, 'ls'),
                 (st_txt, (10, 60 + left.shape[0] + 28), 17,
                  COLORS['A'] if state[k] > 0 else COLORS['B'] if state[k] < 0 else GRAY, True, 'la'),
                 (f'flips counted: {nf}', (10, 60 + left.shape[0] + 56), 19, (100, 240, 100) if nf else (230, 230, 230), True, 'la')]
        if k == len(frames_b) - 1:
            items.append((f'answer: {label}', (10, 60 + left.shape[0] + 88), 21, (100, 240, 100), True, 'la'))
        items.append((f'a flip counts only if the new order holds ≥ 2 consecutive baked frames with margin ≥ {M:.0%}',
                      (10, H_BODY - 12), 13, (190, 190, 190), False, 'ls'))
        out.append(compose(5, text(body, items)))
clip5 = save_clip(5, out, len(out) - 1)

# ---------------------------------------------------------------- stage 6
baked = []
for k, i in enumerate(frames_b):  # identical to render.py: INTER_AREA resize, markers on the first N_MARKED frames
    img = cv2.resize(d['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA)
    if k < N_MARKED:
        for L, j in P.items():
            draw_marker(img, d['uv'][i, j, 0] * sb, d['uv'][i, j, 1] * sb, L, Rb)
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
                       (f'model input: {Wb}×{Hb}, {len(baked)} frames at {in_fps:g} fps (shown at 5 fps, '
                        f'upscaled for display)', (10, H_BODY - 10), 13, (230, 230, 230), False, 'ls')])
    out += [compose(6, disp)] * 2
clip6 = save_clip(6, out, 0)

# ---------------------------------------------------------------- stage 7
body = blank()
thumb = fit(baked[0], 440)
paste(body, thumb, W_OUT - 440 - 16, 90)
items = [('first input frame (marked)', (W_OUT - 16 - 220, 90 + thumb.shape[0] + 20), 13, (190, 190, 190), False, 'mm'),
         ('Question', (24, 60), 16, (170, 170, 170), True, 'la')]
y = 88
for line in textwrap.wrap(Q, 38):
    items.append((line, (24, y), 21, (255, 255, 255), False, 'la'))
    y += 30
y += 20
items.append(('Options', (24, y), 16, (170, 170, 170), True, 'la'))
y += 28
for o in options:
    ok_ = o.startswith(answer + ')') if TPL == 'T4' else o == answer
    items.append((('● ' if ok_ else '○ ') + o, (40, y), 21, (100, 240, 100) if ok_ else (220, 220, 220), ok_, 'la'))
    y += 30
y += 20
items.append(('Answer', (24, y), 16, (170, 170, 170), True, 'la'))
items.append((answer if TPL == 'T5' else f'{answer})  {label}', (24, y + 26), 30, (100, 240, 100), True, 'la'))
if TPL == 'T4':
    gt = 'GT at last frame: ' + ', '.join(f'r_{L} = {r_pt[L][tn]:.2f} m' for L in rankn)
else:
    gt = f'GT: {len(flips_k)} flip(s) at baked frames {[f + 1 for f in flips_k]}'
items.append((f'{gt};  {scene}, frames {t0}–{tn}', (24, H_BODY - 24), 13, (170, 170, 170), False, 'la'))
card = compose(7, text(body, items))
clip7 = save_clip(7, [card] * 50, 0)

# ---------------------------------------------------------------- full video, meta, checks
full = []
for k, c in enumerate([clip0, clip1, clip2, clip3, clip4, clip5, clip6, clip7]):
    full += [title_card(k)] * int(1.5 * FPS_OUT) + c
write_mp4(f'{OUT}/walkthrough_full.mp4', full, FPS_OUT)
print(f'walkthrough_full.mp4: {len(full) / FPS_OUT:.1f} s', flush=True)

# consistency checks: reprojection, depth map vs track, markers only on the first 3 frames, and the label
# recomputed from the depth map (not the tracks) at the pixels the markers point to
check_frames = list(range(t0, tn + 1)) if TPL == 'T4' else [int(i) for i in frames_b]
reproj, dep_err, r_map = {}, {L: [] for L in P}, {L: [] for L in P}
for L, j in P.items():
    proj = d['P_cam'][t0:tn + 1, j] @ K.T
    reproj[L] = float(np.abs(proj[:, :2] / proj[:, 2:3] - d['uv'][t0:tn + 1, j]).max())
for i in check_frames:
    dep = d['depth'](i)
    for L, j in P.items():
        dep_err[L].append(float(depth_agree(dep, d, i, j)))
        u, v = np.round(d['uv'][i, j]).astype(int)
        r_map[L].append(float(dep[v, u] * d['ray_norm'][v, u]))
if TPL == 'T4':
    label_map = ', '.join(sorted(P, key=lambda L: r_map[L][-1]))
else:
    ib = [check_frames.index(int(i)) for i in frames_b]
    _, fa_m, _, _ = flip_trace(np.array(r_map['A'])[ib][:, None], np.array(r_map['B'])[ib][:, None])
    label_map = flip_label(int(fa_m.sum()))
ring_px = []
for k, i in enumerate(frames_b):  # marker present iff the baked frame differs from the plain resize
    plain = cv2.resize(d['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA)
    ring_px.append(int((np.abs(baked[k].astype(int) - plain).sum(-1) > 30).sum()))
checks = dict(reproj_err_px_max=reproj, depth_vs_track_rel_err_max={L: max(v) for L, v in dep_err.items()},
              ring_changed_px_per_baked_frame=ring_px,
              ring_only_on_first_3=bool(all(p > 0 for p in ring_px[:N_MARKED]) and not any(ring_px[N_MARKED:])),
              label_from_depth_map=label_map, label_consistent=label_map == label)
print(json.dumps(checks), flush=True)

meta = dict(dataset='pointodyssey', video_id=scene, fps=fps, resolution=[W, H], template=TPL, n_seconds=N_SEC,
            window=[int(t0), int(tn)], t0=int(t0), tn=int(tn), t0_sec=t0 / fps, tn_sec=tn / fps,
            points={L: dict(track_index=int(j), pixel_t0=[float(x) for x in d['uv'][t0, j]], region_id=pstat[L]['rid'],
                            patch_rel_std_t0=pstat[L]['sd'], r_t0=float(r_pt[L][t0]), r_tn=float(r_pt[L][tn]),
                            r_baked=[round(float(r_pt[L][i]), 4) for i in frames_b]) for L, j in P.items()},
            label=label, question=Q, options=options, answer=answer,
            baked_frames=[int(i) for i in frames_b], baked_resolution=[Wb, Hb], baked_fps=in_fps, ring_radius_px=Rb,
            marker_spacing_px_min=spacing,
            thresholds=dict(m=M, clarity_min_margin=M_SHOW, patch_std=STD_MAX, depth_agree=DEPTH_AGREE, cam_min=CAM_MIN,
                            agent_min=AGENT_MIN, window_stride_s=STEP_SEC, region_min_area=REG_MIN_AREA,
                            region_min_present=REG_MIN_PRESENT, region_max_border=REG_MAX_BORDER, region_bg_area=REG_BG_AREA),
            windows=dict(n=len(win), passed=int(sum(w['passed'] for w in win)),
                         failed_motion=int(sum(not w['motion'] for w in win)),
                         failed_template=int(sum(w['motion'] and not w['tpl'] for w in win))),
            regions={str(i): {k: v for k, v in r.items() if k != 'color'} for i, r in reg.items()},
            stage4_rejected={k: [j for j, _ in v] for k, v in examples.items()}, stage4_missing_reasons=missing4,
            distance_colormap_range_m=[r_lo, r_hi], checks=checks)
if TPL == 'T4':
    meta.update(ranking_t0=rank0, ranking_last=rankn, margins_last=margins_tn, margins_t0=margins_t0)
else:
    meta.update(state_baked=[int(s_) for s_ in state], signed_margin_baked=[round(float(x), 4) for x in sm],
                flip_baked_frames=[f + 1 for f in flips_k], n_flips=len(flips_k))
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
