# Stage-by-stage walkthrough clips for T3 (occlusion) on PointOdyssey.
# Same layout as walkthrough_pointodyssey.py (T1) / walkthrough_po_rank_flip.py (T4, T5): one video, one window,
# points A and B across stages 0-7; every number comes from PointOdyssey GT (depth, camera, 3D tracks, visibility,
# instance masks).
#   Q: "During the video, does point A pass in front of point B?"  Yes / No.
#   positive:      A and B overlap in 2D (<= 1% of image width) on >= 2 consecutive baked frames; there A is visible,
#                  B is hidden and r_A < r_B (margin >= m).
#   hard negative: same overlap, but A hidden behind B (r_A > r_B, A invisible, B visible).
#   easy negative: A never comes within 1% of image width of B.
# Writes to <OUT_ROOT>/walkthrough_T3/item<k>/: stage{k}_<name>.mp4, stage{k}.png, walkthrough_full.mp4, meta.json
#   python walkthrough_po_occlusion.py --select --scenes test/ani2   # scan scenes, write sel/<scene>.json
#   python walkthrough_po_occlusion.py --merge                        # pool sel/*.json, write selection.json (5 picks)
#   python walkthrough_po_occlusion.py --pick k                       # render pick k (0-4)
import argparse, json, os, sys, textwrap
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blk import PO
from common import OUT_ROOT, bake_indices, euclid, in_bounds, load_po, patch_rel_std, po_scenes
from render import COLORS, N_MARKED, baked_size, draw_marker, ring_radius, write_mp4

ap = argparse.ArgumentParser()
ap.add_argument('--select', action='store_true')
ap.add_argument('--merge', action='store_true')
ap.add_argument('--scenes', default='', help='comma list; default all test scenes')
ap.add_argument('--pick', type=int, default=0)
args = ap.parse_args()
TPL = 'T3'
OUT = f'{OUT_ROOT}/walkthrough_{TPL}'
os.makedirs(OUT, exist_ok=True)

FPS_OUT = 10
W_OUT, H_BAN, H_BODY = 960, 64, 540
N_SEC = 4  # spec: 3-5 s
LETTERS = ['A', 'B']
M = 0.10  # relative depth margin |r_i - r_j| / min(r_i, r_j), synthetic data
OVL = 0.01  # overlap radius, fraction of image width
M_SHOW = 0.15  # walkthrough item: larger depth margin during the overlap, for visual clarity
EASY_MIN = 4  # easy negative: closest approach >= EASY_MIN * overlap radius, for visual clarity
VIS_FRONT = 0.8  # clarity: the occluder (easy negative: both points) visible on >= 80% of baked frames
VIS_BACK = 0.7  # clarity: the occluded point visible on >= 70% of baked frames away from the overlap (+-2 frames)
STD_MAX = 0.02  # relative z std in a 5x5 patch
DEPTH_AGREE = 0.02  # |depth map z - track z| / z
STEP_SEC = 0.5  # sliding-window stride
CAM_MIN = 0.2  # m/s, only plotted: T3 needs a moving agent
AGENT_MIN = 0.5  # m/s, mean over the window of the fastest visible moving track
DYN_MIN = 0.05  # m, a track is "on an agent" if its world position ever moves more than this
REG_MIN_AREA = 0.004  # fraction of the image (median over the window)
REG_MIN_PRESENT = 0.9  # fraction of window frames in which the region is visible at all
REG_MAX_BORDER = 0.5  # fraction of window frames in which the region touches the image border
REG_BG_AREA = 0.25  # larger than this at t0 -> scene background, not an object region
PLAN = [('positive', 2), ('hard negative', 2), ('easy negative', 1)]  # spec mix 5/4/1 of 10, halved

GRAY = (150, 150, 150)
PASS_COL, FAIL_COL = (44, 160, 44), (214, 39, 40)
REGION_COLS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]  # no yellow/cyan/magenta
FONT_DIR = f'{matplotlib.get_data_path()}/fonts/ttf'
STAGES = [(0, 'standardize', 'Standardize', 'Common format: RGB, intrinsics, poses, per-pixel distance to camera.'),
          (1, 'geometry', 'Geometry', '3D point tracks + visibility for every frame.'),
          (2, 'clips', 'Clips', 'Slide windows; keep those with a moving agent and an occlusion.'),
          (3, 'regions', 'Regions', 'Pick valid regions from instance masks.'),
          (4, 'points', 'Points', 'Place the query points; keep them only if they pass the checks.'),
          (5, 'labels', 'Labels', 'Compute the answer from GT distance and visibility.'),
          (6, 'render', 'Render', 'Bake frames; mark the points on the first 3 frames only.'),
          (7, 'question', 'Question', 'Final QA item.')]
Q = 'During the video, does point A pass in front of point B?'


# ---------------------------------------------------------------- data + geometry
def load(scene):
    d = load_po(scene)
    a = np.load(f'{PO}/{scene}/anno.npz')
    E = a['extrinsics'].astype(np.float64)
    d['X'] = a['trajs_3d'].astype(np.float32)  # world frame
    d['C'] = np.array([-e[:3, :3].T @ e[:3, 3] for e in E])  # camera centers, world
    d['r'] = euclid(d['P_cam'])
    d['Ks'] = a['intrinsics'].astype(np.float64)  # per frame: some PO scenes zoom (common.load_po keeps only K[0])
    d['ray_cache'] = {}
    d['onscreen'] = d['valid'] & (d['P_cam'][..., 2] > 0.05) & in_bounds(d['uv'], d['W'], d['H'])
    d['occ_cache'] = {}
    return d


def ray_norm(d, i):
    key = tuple(np.round(d['Ks'][i].ravel(), 4))
    if key not in d['ray_cache']:
        uu, vv = np.meshgrid(np.arange(d['W']), np.arange(d['H']))
        d['ray_cache'][key] = np.linalg.norm(np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(d['Ks'][i]).T,
                                             axis=-1).astype(np.float32)
    return d['ray_cache'][key]


def dist_map(d, i):
    z = d['depth'](i)
    r = z * ray_norm(d, i)  # r = z * ||K^-1 [u, v, 1]||, with this frame's K
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


def window_bake(t0, tn):
    return np.array(bake_indices(t0, tn, [t0, tn]))


def occ_keys(d, f):
    """Occlusion pairs at source frame f, as sorted keys front * N + back: front visible, back on screen but hidden,
    2D distance <= overlap radius, r_back >= r_front * (1 + M). Uses every GT track (KD-tree), cached per frame."""
    c = d['occ_cache']
    if f not in c:
        N = d['uv'].shape[1]
        uv = d['uv'][f]
        vis = np.nonzero(d['vis'][f] & d['onscreen'][f])[0]
        hid = np.nonzero(~d['vis'][f] & d['onscreen'][f])[0]
        keys = np.zeros(0, np.int64)
        if len(vis) and len(hid):
            lists = cKDTree(uv[vis]).query_ball_point(uv[hid], d['thr'])
            n = np.array([len(l) for l in lists])
            if n.sum():
                fr = vis[np.concatenate([l for l in lists if l]).astype(int)]
                bk = np.repeat(hid, n)
                ok = d['r'][f, bk] >= d['r'][f, fr] * (1 + M)
                keys = np.unique(fr[ok].astype(np.int64) * N + bk[ok])
        c[f] = keys
    return c[f]


def run_keys(d, fb):
    """Pairs (front, back) that are occlusion pairs on >= 2 consecutive baked frames after the marked ones."""
    K = [occ_keys(d, int(f)) for f in fb]
    out = [np.intersect1d(K[k], K[k + 1]) for k in range(N_MARKED, len(fb) - 1)]
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def windows(d, cam, agent):
    """Motion predicate (a moving agent) + template predicate (some GT track pair has an occlusion event)."""
    L = int(round(N_SEC * d['fps']))
    step = int(round(STEP_SEC * d['fps']))
    out = []
    for t0 in range(0, d['n_frames'] - L, step):
        tn = t0 + L
        c, g = float(cam[t0:tn + 1].mean()), float(agent[t0:tn + 1].mean())
        motion = bool(g >= AGENT_MIN)
        tp = bool(motion and len(run_keys(d, window_bake(t0, tn))))  # the (costly) pair search only if motion passes
        out.append(dict(t0=t0, tn=tn, cam=c, agent=g, motion=motion, tpl=tp, passed=motion and tp))
    return out


def pair_trace(d, fb, ja, jb):
    """Per baked frame: 2D distance, 'close' (both on screen, <= overlap radius), and the occlusion pattern
    'A in front' (A visible, B hidden, r_B >= r_A (1 + M)) / 'B in front' (the reverse)."""
    dist = np.linalg.norm(d['uv'][fb, ja] - d['uv'][fb, jb], axis=-1)
    close = (dist <= d['thr']) & d['onscreen'][fb, ja] & d['onscreen'][fb, jb]
    ra, rb = d['r'][fb, ja], d['r'][fb, jb]
    va, vb = d['vis'][fb, ja], d['vis'][fb, jb]
    a_front = close & va & ~vb & (rb >= ra * (1 + M))
    b_front = close & vb & ~va & (ra >= rb * (1 + M))
    return dist, close, a_front, b_front


def label_of(close, a_front, b_front):
    """Yes if A is in front of B on >= 2 consecutive baked frames and never behind it on a close frame;
    No if never close, or if B is in front on >= 2 consecutive frames and A never in front. Else ambiguous (None)."""
    run = lambda x: bool((x[:-1] & x[1:]).any())
    if not close.any():
        return 'No', 'easy negative'
    if run(a_front) and not b_front.any() and (close == a_front).all():
        return 'Yes', 'positive'
    if run(b_front) and not a_front.any() and (close == b_front).all():
        return 'No', 'hard negative'
    return None, None


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
        d['thr'] = OVL * W
        NT = d['uv'].shape[1]
        cam, agent, dyn = motion_summary(d)
        win = windows(d, cam, agent)
        pad = marker_pad(d)
        print(sc, d['n_frames'], NT, 'agent p50/p90 %.2f/%.2f' % tuple(np.percentile(agent, [50, 90])),
              'windows pass %d/%d' % (sum(w['passed'] for w in win), len(win)), flush=True)
        rng = np.random.default_rng(0)
        sc_c = []
        for w in win:
            t0, tn = w['t0'], w['tn']
            if not w['motion'] or t0 % int(round(fps)):  # search every 1 s
                continue
            fb = window_bake(t0, tn)
            m0 = d['mask'](t0)
            ids = track_ids(d, t0, m0)
            area = np.bincount(m0.ravel(), minlength=int(m0.max()) + 1) / m0.size
            a_id = np.where(ids >= 0, area[np.maximum(ids, 0)], 0)
            obj = (a_id >= REG_MIN_AREA) & (a_id <= REG_BG_AREA)  # an object region (kept-ish)
            surf = a_id >= REG_MIN_AREA  # an object region or the scene background
            # clarity: both points on screen on every baked frame, visible and marker-safe on the marked frames
            okv = d['onscreen'][fb].all(0) & in_bounds(d['uv'][fb], W, H, 20).all(0) \
                & d['vis'][fb[:N_MARKED]].all(0) & in_bounds(d['uv'][fb[:N_MARKED]], W, H, pad).all(0)
            # positives / hard negatives from occlusion events
            keys = run_keys(d, fb)
            if len(keys):
                FR, BK = keys // NT, keys % NT
                ok = okv[FR] & okv[BK] & obj[FR] & surf[BK] & (ids[FR] != ids[BK])
                FR, BK = FR[ok], BK[ok]
                if len(FR) > 4000:
                    s_ = rng.permutation(len(FR))[:4000]
                    FR, BK = FR[s_], BK[s_]
                sp = min_spacing(d['uv'][fb[:N_MARKED]][:, np.stack([FR, BK], 1)]) if len(FR) else np.zeros(0)
                for q in np.nonzero(sp >= pad)[0]:
                    jf, jb = int(FR[q]), int(BK[q])
                    dist, close, a_front, b_front = pair_trace(d, fb, jf, jb)  # A = front here
                    lab, kind = label_of(close, a_front, b_front)
                    if kind != 'positive':
                        continue
                    mg = float(rel_margin(d['r'][fb, jf], d['r'][fb, jb])[close].min())
                    if mg < M_SHOW:
                        continue
                    # clarity: the occluder stays visible, and the occluded point is visible away from the overlap
                    # (not hidden by something else for most of the window)
                    near = np.convolve(close.astype(float), np.ones(5), 'same') > 0  # overlap frames +- 2
                    fvis = float(d['vis'][fb, jf].mean())
                    bvis = float(d['vis'][fb, jb][~near].mean()) if (~near).any() else 0.0
                    if fvis < VIS_FRONT or bvis < VIS_BACK:
                        continue
                    nclose = int(close.sum())
                    base = min(mg, 0.5) + 0.1 * min(nclose, 6) + 0.2 * min(sp[q] / 150, 1) + 0.5 * bvis + 0.3 * fvis
                    # positive: A = the occluder (prefer a moving one); hard negative: A = the hidden point
                    sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=[jf, jb], kind='positive', label='Yes', n_close=nclose,
                                     min_margin_overlap=mg, score=float(base + 0.4 * dyn[jf] + 0.2 * (w['agent'] >= AGENT_MIN)),
                                     cam=w['cam'], agent=w['agent']))
                    sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=[jb, jf], kind='hard negative', label='No', n_close=nclose,
                                     min_margin_overlap=mg, score=float(base + 0.3 * dyn[jb] + 0.2 * dyn[jf]),
                                     cam=w['cam'], agent=w['agent']))
            # easy negatives: A on a moving agent, B elsewhere; A never comes within the overlap radius of B
            JA = rng.permutation(np.nonzero(okv & obj & dyn)[0])[:300]
            JB = rng.permutation(np.nonzero(okv & surf)[0])[:300]
            if len(JA) and len(JB):
                a_, b_ = np.meshgrid(JA, JB, indexing='ij')
                a_, b_ = a_.ravel(), b_.ravel()
                ok = ids[a_] != ids[b_]
                a_, b_ = a_[ok], b_[ok]
                dmin = np.linalg.norm(d['uv'][fb][:, a_] - d['uv'][fb][:, b_], axis=-1).min(0)
                sp = min_spacing(d['uv'][fb[:N_MARKED]][:, np.stack([a_, b_], 1)])
                va, vb = d['vis'][fb][:, a_].mean(0), d['vis'][fb][:, b_].mean(0)
                ok = (dmin >= EASY_MIN * d['thr']) & (dmin <= 150) & (sp >= pad) & (va >= VIS_FRONT) & (vb >= VIS_FRONT)
                sc_e = 0.5 * (1 - np.abs(dmin - 80) / 80) + 0.2 * np.minimum(sp / 150, 1) + 0.2 * (w['agent'] >= AGENT_MIN) \
                    + 0.3 * (va + vb) / 2
                for q in np.argsort(-np.where(ok, sc_e, -1))[:2]:
                    if ok[q]:
                        sc_c.append(dict(scene=sc, t0=t0, tn=tn, j=[int(a_[q]), int(b_[q])], kind='easy negative',
                                         label='No', n_close=0, min_dist_px=float(dmin[q]), score=float(sc_e[q]),
                                         cam=w['cam'], agent=w['agent']))
        # verify the best few per scene and kind: patch std and depth agreement at t0, the occlusion confirmed by the
        # depth map on every overlap frame, the occluder in a kept region and the other point kept or background
        sc_c.sort(key=lambda c: -c['score'])
        kept = {k: 0 for k, _ in PLAN}
        tried = {k: set() for k, _ in PLAN}
        reg_cache = {}
        for c in sc_c:
            kind = c['kind']
            if kept[kind] >= 2 or len(tried[kind]) >= 8 or c['t0'] in tried[kind]:
                continue
            tried[kind].add(c['t0'])
            t0, tn = c['t0'], c['tn']
            ja, jb = c['j']
            dep0 = d['depth'](t0)
            if any(patch_rel_std(dep0, *d['uv'][t0, j]) >= STD_MAX or depth_agree(dep0, d, t0, j) > DEPTH_AGREE
                   for j in c['j']):
                continue
            fb = window_bake(t0, tn)
            if kind != 'easy negative':
                _, close, a_front, b_front = pair_trace(d, fb, ja, jb)
                front, back = (ja, jb) if kind == 'positive' else (jb, ja)
                bad = False
                for k in np.nonzero(close)[0]:
                    dep = d['depth'](int(fb[k]))
                    u, v = np.round(d['uv'][fb[k], back]).astype(int)
                    if depth_agree(dep, d, fb[k], front) > DEPTH_AGREE or dep[v, u] * (1 + M) > d['P_cam'][fb[k], back, 2]:
                        bad = True
                        break
                if bad:
                    continue
            if t0 not in reg_cache:
                reg_cache[t0] = regions(d, t0, tn)
            reg, masks = reg_cache[t0]
            rid = [int(x) for x in track_ids(d, t0, masks[0])[c['j']]]
            lead = rid[0] if kind != 'hard negative' else rid[1]  # the occluder / the moving A must be an object
            other = rid[1] if kind != 'hard negative' else rid[0]
            if lead not in reg or not reg[lead]['keep'] or other not in reg or \
                    not (reg[other]['keep'] or reg[other]['reason'] == 'background'):
                continue
            c['regions'] = rid
            cands.append(c)
            kept[kind] += 1
            print('  cand', json.dumps(c), flush=True)
        del d
        os.makedirs(f'{OUT}/sel', exist_ok=True)
        json.dump([c for c in cands if c['scene'] == sc], open(f'{OUT}/sel/{sc.replace("/", "_")}.json', 'w'), indent=1)
    sys.exit(0)

if args.merge:
    cands = [c for f in sorted(os.listdir(f'{OUT}/sel')) for c in json.load(open(f'{OUT}/sel/{f}'))]
    cands.sort(key=lambda c: -c['score'])
    picks, used = [], set()
    for kind, n in PLAN:  # at most one item per video; fall back to a reused video only if a kind runs out
        pool = [c for c in cands if c['kind'] == kind]
        for c in [c for c in pool if c['scene'] not in used] + [c for c in pool if c['scene'] in used]:
            if sum(p['kind'] == kind for p in picks) >= n:
                break
            if any(p['scene'] == c['scene'] and p['t0'] == c['t0'] for p in picks):
                continue
            picks.append(c)
            used.add(c['scene'])
    json.dump(dict(picks=picks, candidates=cands), open(f'{OUT}/selection.json', 'w'), indent=1)
    print('candidates per kind:', {k: sum(c['kind'] == k for c in cands) for k, _ in PLAN})
    print('picks:', [(p['kind'], p['scene'], p['t0']) for p in picks], flush=True)
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
                    (f'{TPL} — occlusion ({KIND})', (W_OUT - 14, 10), 16, (200, 200, 200), True, 'ra')])


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
    write_mp4(f'{OUT_I}/{name}.mp4', frames, FPS_OUT)
    Image.fromarray(frames[still]).save(f'{OUT_I}/stage{k}.png')
    print(f'{name}.mp4: {len(frames)} frames = {len(frames) / FPS_OUT:.1f} s, still = frame {still}', flush=True)
    return frames


def markers(img, i, R, labels=True, size=20):
    """Each point's marker at source frame i (gray when hidden per GT) and, optionally, its live distance."""
    items = []
    for L, j in P.items():
        u, v = d['uv'][i, j]
        hidden = not d['vis'][i, j]
        draw_marker(img, u, v, L, R, faded=hidden)
        if labels:
            left = u < W - 190
            tx_ = int(u) + R + 4 if left else int(u) - R - 4
            ty = int(v) + R + 22 if v < H - 50 else int(v) - R - 8
            if L == 'B':  # keep the two labels apart when the points overlap
                ty = int(v) - R - 8 if v > 60 else int(v) + R + 44
            items.append((f'{d["r"][i, j]:.2f} m{" (hidden)" if hidden else ""}', (tx_, ty), size,
                          GRAY if hidden else COLORS[L], True, 'ls' if left else 'rs'))
    return items


# ---------------------------------------------------------------- the item
sel = json.load(open(f'{OUT}/selection.json'))
it = sel['picks'][args.pick]
KIND = it['kind']
OUT_I = f'{OUT}/item{args.pick}'
os.makedirs(OUT_I, exist_ok=True)
scene, t0, tn = it['scene'], it['t0'], it['tn']
P = dict(zip(LETTERS, it['j']))  # letter -> track index
d = load(scene)
T, fps, H, W, K = d['n_frames'], d['fps'], d['H'], d['W'], d['K']
d['thr'] = THR = OVL * W
cam, agent, dyn = motion_summary(d)
win = windows(d, cam, agent)
reg, wmasks = regions(d, t0, tn)
frames_b = window_bake(t0, tn)
Wb, Hb, sb = baked_size(H, W)
Rb = ring_radius(Wb)
R_DISP = ring_radius(W)  # ring radius at full (960 px) resolution
WSTEP = 2  # stages 3-4 play every WSTEP-th window frame
r_pt = {L: d['r'][:, j] for L, j in P.items()}
dist_b, close_b, afront_b, bfront_b = pair_trace(d, frames_b, P['A'], P['B'])
label, kind_chk = label_of(close_b, afront_b, bfront_b)
assert kind_chk == KIND, (kind_chk, KIND)
options = ['Yes', 'No']
answer = label
ovl_k = [int(k) for k in np.nonzero(close_b)[0]]
print(scene, 't0', t0, 'tn', tn, 'points', P, 'kind', KIND, 'label', label, 'overlap baked frames', ovl_k, flush=True)

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
                       (f'K: f = {d["Ks"][i][0, 0]:.0f} px, c = ({d["Ks"][i][0, 2]:.0f}, {d["Ks"][i][1, 2]:.0f}) (per frame);  '
                        f'pose: world→camera 4×4 per frame',
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
per = max(2, nwin // 60)  # windows advanced per output frame
steps = list(range(0, nwin, per)) + [nwin - 1]
out = []
A_COL = COLORS['A']
TPL_RULE = (f'some GT track pair overlaps in 2D (≤ {THR:.1f} px = {OVL:.0%} of width) on ≥ 2 consecutive baked frames, '
            f'one hidden behind the other (≥ {M:.0%} farther)')


def stage2_frame(k_done, show_chosen):
    fig, ax = plt.subplots(figsize=(9.4, 1.8), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.2)
    ax.plot(tt, cam, color=(0.2, 0.6, 1.0), lw=1.0, alpha=0.6, label='camera speed')
    ax.plot(tt, agent, color=(1.0, 0.5, 0.1), lw=1.3, label='max agent speed')
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
    why = '' if w['passed'] else ('  (no moving agent)' if not w['motion'] else '  (no occlusion)')
    msg = (f'window {k + 1}/{nwin}: t = {w["t0"] / fps:.1f}–{w["tn"] / fps:.1f} s   agent {w["agent"]:.2f} m/s'
           f'   → {"PASS" if w["passed"] else "FAIL"}{why}' if not final else
           f'chosen window: t = {t0 / fps:.2f}–{tn / fps:.2f} s (frames {t0}–{tn})   {npass}/{nwin} windows pass')
    body = text(body, [(msg, (W_OUT // 2, ytl + 26 + plot.shape[0] + 12), 14,
                        A_COL if final else (PASS_COL if w['passed'] else FAIL_COL), True, 'mm'),
                       (f'{N_SEC} s windows, stride {STEP_SEC} s; pass = agent ≥ {AGENT_MIN} m/s and', (W_OUT // 2, ytl + 26 + plot.shape[0] + 30), 11,
                        (180, 180, 180), False, 'mm'),
                       (TPL_RULE, (W_OUT // 2, ytl + 26 + plot.shape[0] + 46), 11, (180, 180, 180), False, 'mm')])
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
              ('darkened = scene background: allowed only for the occluded / static point', (W - 10, H - 38), 12,
               (170, 170, 170), False, 'ra')]
    out.append(compose(3, text(img, items)))
clip3 = save_clip(3, out, 10)

# ---------------------------------------------------------------- stage 4
# Rejected examples: other candidates tried in place of B (A fixed), first failing check in pipeline order.
ids0 = track_ids(d, t0, wmasks[0])
pad = marker_pad(d)
dep0 = d['depth'](t0)
first = d['vis'][frames_b[:N_MARKED]].all(0)
pool = np.nonzero(d['vis'][t0] & (np.isin(ids0, kept_ids) | np.isin(ids0, bg_ids)) & in_bounds(d['uv'][t0], W, H, pad))[0]
pool = pool[~np.isin(pool, list(P.values()))]
rng = np.random.default_rng(1)
rng.shuffle(pool)
pool = pool[:4000]
if KIND == 'easy negative':
    fails = {'not visible on marked frames': [], 'depth variance': [], 'too close to A': [], f'comes within {THR:.0f} px of A': []}
else:
    fails = {'not visible on marked frames': [], 'depth variance': [], 'too close to A': [], 'never overlaps A': [],
             'overlap, depth order unclear': [], 'wrong occlusion type': []}
dist_p = np.linalg.norm(d['uv'][frames_b][:, pool] - d['uv'][frames_b, P['A']][:, None], axis=-1)
close_p = (dist_p <= THR) & d['onscreen'][frames_b][:, pool] & d['onscreen'][frames_b, P['A']][:, None]
for q_, j in enumerate(pool):
    near = np.linalg.norm(d['uv'][frames_b[:N_MARKED], j] - d['uv'][frames_b[:N_MARKED], P['A']], axis=-1).min()
    if not first[j]:
        fails['not visible on marked frames'].append((int(j), 'occluded on a marked frame'))
        continue
    sd = patch_rel_std(dep0, *d['uv'][t0, j])
    if sd >= STD_MAX:
        if np.isfinite(sd):
            fails['depth variance'].append((int(j), f'σ/μ = {sd:.0%}'))
        continue
    if near < pad:
        fails['too close to A'].append((int(j), f'{near:.0f} px < {pad:.0f} px'))
        continue
    if KIND == 'easy negative':
        if close_p[:, q_].any():
            fails[f'comes within {THR:.0f} px of A'].append((int(j), f'closest {dist_p[:, q_].min():.0f} px'))
        continue
    cl = close_p[:, q_]
    if not (cl[:-1] & cl[1:]).any():
        fails['never overlaps A'].append((int(j), f'closest {dist_p[:, q_].min():.0f} px'))
        continue
    _, c_, af, bf = pair_trace(d, frames_b, P['A'], int(j))
    lab_q, kind_q = label_of(c_, af, bf)
    if lab_q is None:
        fails['overlap, depth order unclear'].append((int(j), f'margin {rel_margin(r_pt["A"][frames_b], d["r"][frames_b, j])[c_].min():.0%}'))
    elif kind_q != KIND:
        fails['wrong occlusion type'].append((int(j), f'would be a {kind_q}'))
examples = {k: [] for k in fails}
placed = [d['uv'][t0, j] for j in P.values()]
for why in sorted(fails, key=lambda k: len(fails[k])):  # rarest reason first, so every reason gets tag space
    for j, note in fails[why]:
        if len(examples[why]) >= 2:
            break
        if any(abs(d['uv'][t0, j, 0] - p[0]) < 170 and abs(d['uv'][t0, j, 1] - p[1]) < 50 for p in placed):
            continue
        if not (60 < d['uv'][t0, j, 1] < H - 130 and d['uv'][t0, j, 0] < W - 190):
            continue
        examples[why].append((j, note))
        placed.append(d['uv'][t0, j])
print('stage 4 rejected examples:', examples, flush=True)
missing4 = [k for k, v in examples.items() if not v]
pstat = {L: dict(sd=patch_rel_std(dep0, *d['uv'][t0, j]), rid=int(ids0[j]),
                 region='background' if int(ids0[j]) in bg_ids else f'region {int(ids0[j])}') for L, j in P.items()}
spacing = float(np.linalg.norm(d['uv'][frames_b[:N_MARKED], P['A']] - d['uv'][frames_b[:N_MARKED], P['B']], axis=-1).min())


def legend_box(img, lines):
    """Bottom-left panel listing each point's check values."""
    h = 22 * len(lines) + 12
    y0 = H - 26 - h
    sub = img[y0:H - 26, 0:520].astype(np.float32)
    img[y0:H - 26, 0:520] = (sub * 0.35 + 18 * 0.65).astype(np.uint8)
    return [(s, (10, y0 + 8 + 22 * n), 14, col, bold, 'la') for n, (s, col, bold) in enumerate(lines)]


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
        for L, j in P.items():
            draw_marker(img, *d['uv'][t0, j], L, R_DISP)
        lines = [('✓ chosen points pass all checks', (120, 230, 120), True)]
        for L, j in P.items():
            lines.append((f'{L}: {pstat[L]["region"]}{" (moving)" if dyn[j] else ""}, σ/μ = {pstat[L]["sd"]:.2%}, '
                          f'visible at t0, r = {r_pt[L][t0]:.2f} m', COLORS[L], False))
        if KIND == 'easy negative':
            lines.append((f'closest 2D approach {dist_b.min():.0f} px > {THR:.1f} px: never overlap', (230, 230, 230), False))
        else:
            lines.append((f'2D overlap (≤ {THR:.1f} px) on baked frames {[k + 1 for k in ovl_k]}', (230, 230, 230), False))
        lines.append((f'min marker spacing {spacing:.0f} px ≥ {pad:.0f} px', (230, 230, 230), False))
        items += legend_box(img, lines)
        items.append(('Candidate points at t0 (A fixed; gray = rejected candidates for B)', (10, 16), 15, (240, 240, 240), True, 'la'))
    else:
        for L, j in P.items():
            trail(img, d['uv'][t0:i + 1, j], COLORS[L], 2)
        items += markers(img, i, R_DISP, labels=False)
        items.append((f'GT tracks   t0 + {q / fps:.2f} s   (gray ring = hidden per GT visibility)', (10, 16), 15, (240, 240, 240), True, 'la'))
    cv2.rectangle(img, (0, H - 26), (W, H), (18, 18, 18), -1)
    rule = (f'checks: visible on the first {N_MARKED} frames;  5×5 depth σ/μ < {STD_MAX:.0%};  markers ≥ 3 ring radii apart;  '
            + (f'never within {THR:.1f} px' if KIND == 'easy negative' else
               f'overlap ≤ {THR:.1f} px on ≥ 2 consecutive baked frames, depth margin ≥ {M:.0%}'))
    items.append((rule, (10, H - 8), 11, (200, 200, 200), False, 'ls'))
    out.append(compose(4, text(img, items)))
clip4 = save_clip(4, out, 5)

# ---------------------------------------------------------------- stage 5
out = []
tw = np.arange(tn - t0 + 1) / fps
tb = (frames_b - t0) / fps
rw = {L: r_pt[L][t0:tn + 1] for L in P}
vw = {L: d['vis'][t0:tn + 1, j] for L, j in P.items()}
allr = np.concatenate(list(rw.values()))
dmax = float(max(dist_b.max(), 3 * THR))


def stage5_plot(k):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.7, 4.5), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.1]))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.11, hspace=0.12)
    dark_axes(fig, [ax, bx])
    q = frames_b[k] - t0
    ax.axhspan(0, THR, color=(0.3, 1.0, 0.3), alpha=0.2, lw=0)
    ax.text(0.02, THR, f'overlap ≤ {THR:.1f} px', color=(0.6, 1.0, 0.6), fontsize=7, va='bottom')
    ax.plot(tb[:k + 1], dist_b[:k + 1], 'o-', color=(0.9, 0.9, 0.9), ms=3, lw=1.2)
    for kk in ovl_k:
        if kk <= k:
            ax.plot(tb[kk], dist_b[kk], 'o', color=(0.3, 1.0, 0.3), ms=6, mec='k')
    ax.set_yscale('symlog', linthresh=THR)
    ax.set_ylim(0, dmax * 1.3)
    ax.set_ylabel('|A − B| in image (px)', color='w', fontsize=8)
    ax.set_title('2D distance; dots = baked frames', color='w', fontsize=10)
    for L in P:  # r over time: solid while visible, dotted while hidden
        rv = np.where(vw[L][:q + 1], rw[L][:q + 1], np.nan)
        rh = np.where(~vw[L][:q + 1], rw[L][:q + 1], np.nan)
        bx.plot(tw[:q + 1], rv, color=mcol(L), lw=2.2, label=f'{L} visible')
        bx.plot(tw[:q + 1], rh, color=mcol(L), lw=1.5, ls=':', label=f'{L} hidden')
    for kk in ovl_k:
        if kk <= k:
            bx.axvspan(tb[kk] - 0.5 * (tb[1] - tb[0]), tb[kk] + 0.5 * (tb[1] - tb[0]), color=(0.3, 1.0, 0.3), alpha=0.2, lw=0)
    bx.set_xlim(-0.1, tw[-1] + 0.1)
    bx.set_ylim(allr.min() - 0.1 * np.ptp(allr) - 0.05, allr.max() + 0.1 * np.ptp(allr) + 0.05)
    bx.set_ylabel('r (m)', color='w')
    bx.set_xlabel('time since t0 (s)', color='w')
    bx.legend(loc='upper right', fontsize=7, facecolor=(0.1, 0.1, 0.1), labelcolor='w', framealpha=0.8, ncol=2)
    return fig_to_array(fig)


for k in [x for k in range(len(frames_b)) for x in [k] * 3] + [len(frames_b) - 1] * 22:
    i = frames_b[k]
    img = d['image'](i)
    for L, j in P.items():
        trail(img, d['uv'][t0:i + 1, j], COLORS[L], 2)
    if close_b[k]:
        circle(img, *d['uv'][i, P['B']], int(round(THR)), (100, 240, 100))
    img = text(img, markers(img, i, R_DISP, True, 22))
    left = fit(img, 470)
    body = blank()
    paste(body, left, 10, 60)
    plot = stage5_plot(k)
    paste(body, plot, W_OUT - plot.shape[1] - 6, (H_BODY - plot.shape[0]) // 2)
    st = []
    for L, j in P.items():
        st.append(f'{L} {"visible" if d["vis"][i, j] else "hidden"}')
    if close_b[k]:
        who = 'A in front of B' if afront_b[k] else 'B in front of A'
        st_txt, st_col = f'overlap ({dist_b[k]:.1f} px): {who}', (100, 240, 100)
    else:
        st_txt, st_col = f'no overlap ({dist_b[k]:.0f} px apart)', (230, 230, 230)
    seen = [kk for kk in ovl_k if kk <= k]
    items = [(f'baked frame {k + 1}/{len(frames_b)}   (source {i}, t0 + {(i - t0) / fps:.2f} s)', (10, 50), 15, (230, 230, 230), False, 'ls'),
             (st_txt, (10, 60 + left.shape[0] + 26), 17, st_col, True, 'la'),
             ('   '.join(st), (10, 60 + left.shape[0] + 52), 15, (210, 210, 210), False, 'la'),
             (f'overlap frames so far: {len(seen)}', (10, 60 + left.shape[0] + 76), 15, (210, 210, 210), False, 'la')]
    if k == len(frames_b) - 1:
        why = {'positive': 'A passes in front of B', 'hard negative': 'A passes behind B',
               'easy negative': 'A never overlaps B'}[KIND]
        items.append((f'answer: {label}   ({why})', (10, 60 + left.shape[0] + 106), 21, (100, 240, 100), True, 'la'))
    items.append((f'Yes iff on ≥ 2 consecutive overlap frames A is visible, B hidden and r_A < r_B by ≥ {M:.0%}',
                  (10, H_BODY - 20), 13, (190, 190, 190), False, 'la'))
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
write_mp4(f'{OUT_I}/stage6_model_input.mp4', baked, in_fps)  # the exact model input, native size and rate
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
    ok_ = o == answer
    items.append((('● ' if ok_ else '○ ') + o, (40, y), 21, (100, 240, 100) if ok_ else (220, 220, 220), ok_, 'la'))
    y += 30
y += 20
items.append(('Answer', (24, y), 16, (170, 170, 170), True, 'la'))
items.append((f'{answer}   ({KIND})', (24, y + 26), 30, (100, 240, 100), True, 'la'))
if KIND == 'easy negative':
    gt = f'GT: closest 2D approach {dist_b.min():.0f} px > {THR:.1f} px'
else:
    k_ = ovl_k[0]
    gt = (f'GT: overlap on baked frames {[k + 1 for k in ovl_k]}; at frame {k_ + 1}: r_A = {r_pt["A"][frames_b[k_]]:.2f} m '
          f'({"visible" if d["vis"][frames_b[k_], P["A"]] else "hidden"}), r_B = {r_pt["B"][frames_b[k_]]:.2f} m '
          f'({"visible" if d["vis"][frames_b[k_], P["B"]] else "hidden"})')
items.append((gt, (24, H_BODY - 44), 13, (170, 170, 170), False, 'la'))
items.append((f'{scene}, frames {t0}–{tn}', (24, H_BODY - 24), 13, (170, 170, 170), False, 'la'))
card = compose(7, text(body, items))
clip7 = save_clip(7, [card] * 50, 0)

# ---------------------------------------------------------------- full video, meta, checks
full = []
for k, c in enumerate([clip0, clip1, clip2, clip3, clip4, clip5, clip6, clip7]):
    full += [title_card(k)] * int(1.5 * FPS_OUT) + c
write_mp4(f'{OUT_I}/walkthrough_full.mp4', full, FPS_OUT)
print(f'walkthrough_full.mp4: {len(full) / FPS_OUT:.1f} s', flush=True)

# consistency checks: reprojection; the depth map at t0; on every overlap frame, which point the depth map sees
# (label recomputed from the depth map, not from the GT visibility flags); markers only on the first 3 frames
reproj = {}
for L, j in P.items():
    proj = np.einsum('tij,tj->ti', d['Ks'][t0:tn + 1], d['P_cam'][t0:tn + 1, j])  # this frame's K
    reproj[L] = float(np.abs(proj[:, :2] / proj[:, 2:3] - d['uv'][t0:tn + 1, j]).max())
dep_t0 = {L: float(depth_agree(dep0, d, t0, j)) for L, j in P.items()}
seen_by_map = []
for k in ovl_k:
    dep = d['depth'](int(frames_b[k]))
    who = [L for L, j in P.items() if depth_agree(dep, d, frames_b[k], j) <= DEPTH_AGREE]
    seen_by_map.append(''.join(who))
if KIND == 'easy negative':
    label_map = 'No' if not close_b.any() else '?'
else:
    label_map = 'Yes' if seen_by_map and all(s == 'A' for s in seen_by_map) else \
        'No' if seen_by_map and all(s == 'B' for s in seen_by_map) else '?'
ring_px = []
for k, i in enumerate(frames_b):  # marker present iff the baked frame differs from the plain resize
    plain = cv2.resize(d['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA)
    ring_px.append(int((np.abs(baked[k].astype(int) - plain).sum(-1) > 30).sum()))
checks = dict(reproj_err_px_max=reproj, depth_vs_track_rel_err_t0=dep_t0,
              depth_map_visible_point_per_overlap_frame=seen_by_map,
              ring_changed_px_per_baked_frame=ring_px,
              ring_only_on_first_3=bool(all(p > 0 for p in ring_px[:N_MARKED]) and not any(ring_px[N_MARKED:])),
              label_from_depth_map=label_map, label_consistent=label_map == label)
print(json.dumps(checks), flush=True)

meta = dict(dataset='pointodyssey', video_id=scene, fps=fps, resolution=[W, H], template=TPL, kind=KIND, n_seconds=N_SEC,
            window=[int(t0), int(tn)], t0=int(t0), tn=int(tn), t0_sec=t0 / fps, tn_sec=tn / fps,
            points={L: dict(track_index=int(j), pixel_t0=[float(x) for x in d['uv'][t0, j]], region_id=pstat[L]['rid'],
                            region=pstat[L]['region'], on_moving_agent=bool(dyn[j]), patch_rel_std_t0=pstat[L]['sd'],
                            r_baked=[round(float(r_pt[L][i]), 4) for i in frames_b],
                            vis_baked=[bool(d['vis'][i, j]) for i in frames_b]) for L, j in P.items()},
            dist_2d_baked_px=[round(float(x), 2) for x in dist_b], overlap_radius_px=THR,
            overlap_baked_frames=[k + 1 for k in ovl_k],
            label=label, question=Q, options=options, answer=answer,
            baked_frames=[int(i) for i in frames_b], baked_resolution=[Wb, Hb], baked_fps=in_fps, ring_radius_px=Rb,
            marker_spacing_px_min=spacing,
            thresholds=dict(m=M, overlap_frac_width=OVL, clarity_min_margin=M_SHOW, easy_min_dist_x_radius=EASY_MIN,
                            clarity_vis_front=VIS_FRONT, clarity_vis_back=VIS_BACK,
                            patch_std=STD_MAX, depth_agree=DEPTH_AGREE, agent_min=AGENT_MIN, window_stride_s=STEP_SEC,
                            region_min_area=REG_MIN_AREA, region_min_present=REG_MIN_PRESENT,
                            region_max_border=REG_MAX_BORDER, region_bg_area=REG_BG_AREA),
            windows=dict(n=len(win), passed=int(sum(w['passed'] for w in win)),
                         failed_motion=int(sum(not w['motion'] for w in win)),
                         failed_template=int(sum(w['motion'] and not w['tpl'] for w in win))),
            regions={str(i): {k: v for k, v in r.items() if k != 'color'} for i, r in reg.items()},
            stage4_rejected={k: [j for j, _ in v] for k, v in examples.items()}, stage4_missing_reasons=missing4,
            distance_colormap_range_m=[r_lo, r_hi], checks=checks)
json.dump(meta, open(f'{OUT_I}/meta.json', 'w'), indent=1)
