# Production runner for the point-level depth video QA templates (T1, T3, T4, T5, T6).
# Reuses the validated walkthrough code: the preamble (constants + helpers) and the selection block of each
# walkthrough script are exec'd with OUT redirected to the work dir and some walkthrough-only clarity thresholds relaxed.
# Items are rendered with render.render_item (32 baked frames, markers on the first 3) and written to
#   /shared/rsaas/joeyw3/pvideo/<T>/<T>_NNN.mp4   (the model input video)
#   /shared/rsaas/joeyw3/pvideo/<T>/<T>_NNN.json  (question, options, answer, points, frames, provenance)
#   /shared/rsaas/joeyw3/pvideo/<T>/overlay/<T>_NNN_overlay.mp4  (GT overlay for checking)
# Numbering continues after the highest existing <T>_NNN, and a video already used by an existing item is not reused.
#   python run.py select T3 train/ani            # candidates for one scene/drive -> _work/T3/sel/...
#   python run.py assign T3 --n 20 --per_job 5   # pick, number, write _work/T3/assign_<stamp>.json + job list
#   python run.py render T3 <assign.json> 8,9,10  # render those numbers (skips numbers already on disk)
import argparse, glob, json, os, re, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
PROD = '/shared/rsaas/joeyw3/pvideo'
WORK = f'{PROD}/_work'
SCRIPT = {'T1': 'walkthrough_pointodyssey.py', 'T3': 'walkthrough_po_occlusion.py', 'T4': 'walkthrough_po_rank_flip.py',
          'T5': 'walkthrough_po_rank_flip.py', 'T6': 'walkthrough_kitti_event.py'}
SEL_MARK = '# ---------------------------------------------------------------- selection'
DRAW_MARK = '# ---------------------------------------------------------------- drawing helpers'
T1_SEC = {0: 1, 1: 2, 2: 3}  # T1 horizon n (s), chosen per scene by a stable hash


def stable_hash(s):
    return sum((i + 1) * ord(c) for i, c in enumerate(s))


def safe(name):
    return name.replace('/', '_')


def load_ns(tpl, argv):
    """exec the walkthrough preamble (everything before the selection block) and return its namespace + the
    selection block source."""
    src = open(f'{SRC}/{SCRIPT[tpl]}').read()
    pre, rest = src.split(SEL_MARK, 1)
    sel = rest.split(DRAW_MARK, 1)[0]
    old = sys.argv
    sys.argv = ['x'] + argv
    ns = {'__name__': f'wt_{tpl}', '__file__': f'{SRC}/{SCRIPT[tpl]}'}
    try:
        exec(compile(pre, SCRIPT[tpl], 'exec'), ns)
    finally:
        sys.argv = old
    return ns, sel


def base_argv(tpl):
    return ['--template', tpl] if tpl in ('T4', 'T5') else []


# ---------------------------------------------------------------- selection
def patch_select(tpl, ns, sel, name):
    """Redirect outputs and relax walkthrough-only clarity rules (spec rules are kept)."""
    if tpl == 'T1':
        ns['N_SEC'] = T1_SEC[stable_hash(name) % 3]
        ns['M_SHOW'] = 0.15  # spec m = 0.10; a bit larger so the change is visible
        ns['OUT'] = f'{WORK}/T1/sel/{safe(name)}'
        sel = sel.replace('if kept >= 3 or len(seen) >= 12:', 'if kept >= 4 or len(seen) >= 16:')
        # verify Closer and Farther candidates alternately so both labels survive per scene
        sel = sel.replace("sc_c.sort(key=lambda c: -c['score'])",
                          "sc_c.sort(key=lambda c: -c['score']); _cl = [c for c in sc_c if c['rel'] < 0]; "
                          "_fa = [c for c in sc_c if c['rel'] > 0]; "
                          "sc_c = [x for p in zip(_cl, _fa) for x in p] + _cl[len(_fa):] + _fa[len(_cl):]")
    elif tpl == 'T3':
        ns['OUT'] = f'{WORK}/T3'
        ns['M_SHOW'] = 0.12
        sel = sel.replace("if kept[kind] >= 2 or len(tried[kind]) >= 8", "if kept[kind] >= 2 or len(tried[kind]) >= 12")
    elif tpl == 'T4':
        ns['OUT'] = f'{WORK}/T4/sel/{safe(name)}'
        ns['M_SHOW'] = 0.12
        # keep both changed and unchanged rankings (spec: >= half of the items change), and store the flag
        sel = sel.replace('ok = (mn >= M_SHOW) & (m0v >= M) & changed & (sp >= pad) & (ndist >= 2)',
                          'ok = (mn >= M_SHOW) & (m0v >= M) & (sp >= pad) & (ndist >= 2)')
        sel = sel.replace("min_margin_tn=float(mn[q]),", "min_margin_tn=float(mn[q]), changed=bool(changed[q]),")
        sel = sel.replace("score = np.minimum(mn, 0.5)", "score = 0.3 * changed + np.minimum(mn, 0.5)")
        sel = sel.replace('if kept >= 3 or len(reg_cache) >= 12:', 'if kept >= 4 or len(reg_cache) >= 16:')
    elif tpl == 'T5':
        ns['OUT'] = f'{WORK}/T5/sel/{safe(name)}'
        ns['M_SHOW'] = 0.12
        # allow 0-flip pairs ("Never"), slightly shorter clear runs
        sel = sel.replace('ok = init_ok & (nf >= 1) & (sp >= pad)', 'ok = init_ok & (nf >= 0) & (sp >= pad)')
        sel = sel.replace('ok &= (peak >= M_SHOW) & (clear >= 0.7)', 'ok &= (peak >= M_SHOW) & (clear >= 0.6)')
        sel = sel.replace('if np.diff(np.r_[0, fk, len(fb)]).min() < 4:', 'if np.diff(np.r_[0, fk, len(fb)]).min() < 3:')
        # per window keep the best of each label so "Never" candidates survive next to flipping ones
        sel = sel.replace('if not ok[q] or n_ok >= 3:', 'if not ok[q] or n_ok >= 6:')
        sel = sel.replace('if kept >= 3 or len(reg_cache) >= 12:', 'if kept >= 6 or len(reg_cache) >= 20:')
        sel = sel.replace('            if t0 in used:\n                continue\n',
                          "            if (t0, c['label']) in used:\n                continue\n")
        sel = sel.replace('            used.add(t0)\n', "            used.add((t0, c['label']))\n")
    elif tpl == 'T6':
        ns['OUT'] = f'{WORK}/T6'
    os.makedirs(ns['OUT'], exist_ok=True)
    return sel


def cmd_select(tpl, name):
    flag = '--drives' if tpl == 'T6' else '--scenes'
    ns, sel = load_ns(tpl, base_argv(tpl) + ['--select', flag, name])
    sel = patch_select(tpl, ns, sel, name)
    t = time.time()
    try:
        exec(compile(sel, SCRIPT[tpl] + ':select', 'exec'), ns)
    except SystemExit:
        pass
    # normalise to one file per scene: _work/<T>/cands/<scene>.json
    if tpl in ('T1', 'T4', 'T5'):
        cands = json.load(open(f"{ns['OUT']}/selection.json"))
        if tpl == 'T1':
            for c in cands:
                c['n_sec'] = ns['N_SEC']
                c['label'] = 'Farther' if c['rel'] > 0 else 'Closer'
        if tpl == 'T4':
            for c in cands:
                c['label'] = 'changed' if c.get('changed') else 'same'
    elif tpl == 'T3':
        cands = json.load(open(f"{ns['OUT']}/sel/{safe(name)}.json"))
    else:
        cands = json.load(open(f"{ns['OUT']}/sel/{name}.json"))
        for c in cands:
            c['scene'] = c['drive']
            c['label'] = 'changed' if c['changed'] else 'same'
    os.makedirs(f'{WORK}/{tpl}/cands', exist_ok=True)
    json.dump(cands, open(f'{WORK}/{tpl}/cands/{safe(name)}.json', 'w'), indent=1)
    print(f'SELECT DONE {tpl} {name}: {len(cands)} candidates in {time.time() - t:.0f} s', flush=True)


# ---------------------------------------------------------------- numbering
def existing(tpl):
    """{num: video_id} of the items already on disk (an item counts once its .json exists)."""
    out = {}
    for f in glob.glob(f'{PROD}/{tpl}/{tpl}_*.json'):
        m = re.match(rf'{tpl}_(\d+)\.json$', os.path.basename(f))
        if m:
            try:
                out[int(m.group(1))] = json.load(open(f))['video_id']
            except Exception:
                out[int(m.group(1))] = None
    return out


def next_num(tpl):
    nums = list(existing(tpl))
    for f in glob.glob(f'{PROD}/{tpl}/{tpl}_*.mp4'):  # also count a video whose json is missing
        m = re.match(rf'{tpl}_(\d+)\.mp4$', os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


PLAN = {'T1': [('Closer', 0.5), ('Farther', 0.5)],
        'T3': [('positive', 0.5), ('hard negative', 0.4), ('easy negative', 0.1)],
        'T4': [('changed', 0.5), ('same', 0.5)],
        'T5': [('Never', 0.3), ('Once', 0.35), ('More than once', 0.35)],
        'T6': [('same', 1.0)]}


def kind_of(tpl, c):
    return c['kind'] if tpl == 'T3' else c['label']


def cmd_assign(tpl, n, per_job, exclude_pending, only_kind=''):
    files = sorted(glob.glob(f'{WORK}/{tpl}/cands/*.json'))
    cands = [c for f in files for c in json.load(open(f))]
    cands.sort(key=lambda c: -c['score'])
    used = {v for v in existing(tpl).values() if v}
    if exclude_pending:  # videos assigned earlier but not rendered yet
        for f in glob.glob(f'{WORK}/{tpl}/assign_*.json'):
            used |= {a['cand']['scene'] for a in json.load(open(f))['items']}
    pool = [c for c in cands if c['scene'] not in used and (not only_kind or kind_of(tpl, c) == only_kind)]
    picks, vids = [], set()
    quota = {only_kind: n} if only_kind else {k: int(round(f * n)) for k, f in PLAN[tpl]}
    for k, q in quota.items():  # by kind, best first, one item per video
        for c in pool:
            if sum(kind_of(tpl, p) == k for p in picks) >= q:
                break
            if kind_of(tpl, c) == k and c['scene'] not in vids:
                picks.append(c)
                vids.add(c['scene'])
    for c in pool:  # fill up with any kind
        if len(picks) >= n:
            break
        if c['scene'] not in vids:
            picks.append(c)
            vids.add(c['scene'])
    rng = np.random.default_rng(len(used) + 7)
    order = rng.permutation(len(picks))  # mix kinds over the numbering
    start = next_num(tpl)
    for f in glob.glob(f'{WORK}/{tpl}/assign_*.json'):  # never reuse a number already handed out
        start = max([start] + [a['num'] + 1 for a in json.load(open(f))['items']])
    items = [dict(num=start + i, cand=picks[k]) for i, k in enumerate(order)]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = f'{WORK}/{tpl}/assign_{stamp}.json'
    json.dump(dict(template=tpl, items=items), open(path, 'w'), indent=1)
    nums = [it['num'] for it in items]
    with open(f'{WORK}/{tpl}/render_jobs_{stamp}.txt', 'w') as f:
        for i in range(0, len(nums), per_job):
            f.write(f"{tpl} {path} {','.join(str(x) for x in nums[i:i + per_job])}\n")
    print(f'{tpl}: {len(cands)} candidates from {len(files)} videos, {len(pool)} on unused videos; '
          f'assigned {len(items)} -> numbers {nums[:1]}..{nums[-1:]}; kinds',
          {k: sum(kind_of(tpl, p) == k for p in picks) for k, _ in PLAN[tpl]})
    print('ASSIGN', path, f'{WORK}/{tpl}/render_jobs_{stamp}.txt')


# ---------------------------------------------------------------- rendering
def write_item(tpl, num, data, frames, pts, label_frames, q, opts, ans, extra):
    from render import render_item
    name = f'{tpl}_{num:03d}'
    dst = f'{PROD}/{tpl}'
    os.makedirs(f'{dst}/overlay', exist_ok=True)
    tmp = f'{WORK}/{tpl}/tmp/{name}'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    render_item(tmp, data, [int(x) for x in frames], pts, [int(x) for x in label_frames], q, opts, ans,
                dict(template=tpl, item_id=name, **extra))
    shutil.move(f'{tmp}/overlay.mp4', f'{dst}/overlay/{name}_overlay.mp4')
    shutil.move(f'{tmp}/input.mp4', f'{dst}/{name}.mp4')
    meta = json.load(open(f'{tmp}/meta.json'))
    meta['video_file'] = f'{name}.mp4'
    json.dump(meta, open(f'{dst}/{name}.json.tmp', 'w'), indent=1)
    os.replace(f'{dst}/{name}.json.tmp', f'{dst}/{name}.json')  # the json marks the item complete
    shutil.rmtree(tmp)
    print('WROTE', f'{dst}/{name}.mp4', '|', q, opts, '->', ans, flush=True)


def jlist(x):
    return [int(v) if isinstance(v, (np.integer,)) else v for v in x]


def render_po(tpl, ns, d, c, num):
    t0, tn = c['t0'], c['tn']
    js = c['j'] if isinstance(c['j'], list) else [c['j']]
    letters = ['A', 'B', 'C'][:len(js)]
    if tpl == 'T1':
        from common import bake_indices
        fb = np.array(bake_indices(t0, tn, [t0, tn]))
    else:
        fb = ns['window_bake'](t0, tn)
    pts = {L: dict(uv=d['uv'][fb, j], r=d['r'][fb, j], vis=d['vis'][fb, j]) for L, j in zip(letters, js)}
    extra = dict(t0=int(t0), tn=int(tn), track_indices=dict(zip(letters, [int(j) for j in js])),
                 selection=c, source_script=SCRIPT[tpl])
    if tpl == 'T1':
        n = c['n_sec']
        q = ns['Q'].format(n=n)
        rel = (d['r'][tn, js[0]] - d['r'][t0, js[0]]) / d['r'][t0, js[0]]
        ans = 'Farther' if rel > 0 else 'Closer'
        assert ans == c['label']
        return fb, pts, [t0, tn], q, ['Closer', 'Farther'], ans, dict(extra, n_seconds=n, rel_change=float(rel))
    if tpl == 'T3':
        dist, close, a_front, b_front = ns['pair_trace'](d, fb, js[0], js[1])
        lab, kind = ns['label_of'](close, a_front, b_front)
        assert lab == c['label'] and kind == c['kind'], (lab, kind, c)
        lf = fb[close] if close.any() else [t0, tn]
        return fb, pts, lf, ns['Q'], ['Yes', 'No'], lab, dict(extra, kind=kind, overlap_baked_frames=
                                                                 [int(k) + 1 for k in np.nonzero(close)[0]])
    if tpl == 'T4':
        r_tn = {L: float(d['r'][tn, j]) for L, j in zip(letters, js)}
        order = sorted(letters, key=lambda L: r_tn[L])
        perms = [order] + [[order[1], order[0], order[2]], [order[0], order[2], order[1]], [order[2], order[1], order[0]]]
        rng = np.random.default_rng(num)
        perms = [perms[k] for k in rng.permutation(4)]
        opts = [f"{'abcd'[k]}) {', '.join(p)}" for k, p in enumerate(perms)]
        ans = opts[perms.index(order)]
        r_t0 = {L: float(d['r'][t0, j]) for L, j in zip(letters, js)}
        return fb, pts, [tn], ns['Q'], opts, ans, dict(extra, ranking_tn=order, ranking_t0=sorted(letters, key=lambda L: r_t0[L]),
                                                      r_t0=r_t0, r_tn=r_tn)
    if tpl == 'T5':
        R = d['r'][fb][:, js]
        state, flip_at, init_ok, sm = ns['flip_trace'](R[:, :1], R[:, 1:])
        nf = int(flip_at.sum())
        ans = ns['flip_label'](nf)
        assert init_ok[0] and ans == c['label'], (nf, c)
        lf = [int(fb[k]) for k in np.nonzero(flip_at[:, 0])[0]] or [int(t0), int(tn)]
        return fb, pts, lf, ns['Q'], ['Never', 'Once', 'More than once'], ans, \
            dict(extra, n_flips=nf, flip_baked_frames=[int(k) + 1 for k in np.nonzero(flip_at[:, 0])[0]])


def render_kitti(ns, c, num):
    from common import bake_indices
    d = ns['load'](c['drive'])
    t0, te, tn = c['t0'], c['te'], c['tn']
    fb = np.array(bake_indices(t0, tn, [t0, te, tn]))
    Xv = ns['lidar'](d, t0)['Xv'][c['j']]  # the 3 chosen t0 returns, in velodyne coordinates
    uv, vis, r = [], [], []
    for t in fb:
        Xc = ns['xform'](ns['velo_t0_to_cam'](d, t0, int(t)), Xv)
        v_, uv_, on_ = ns['zbuffer'](d, int(t), Xc)
        uv.append(uv_), vis.append(v_), r.append(np.linalg.norm(Xc, axis=1))
    uv, vis, r = np.array(uv), np.array(vis), np.array(r)
    # after t_e the car can pass a point and it leaves the view (z <= 0 or off-image): hold its last on-screen pixel
    # (drawn faded, vis is False there); markers on the 3 marked frames are always on screen (checked at selection)
    for p in range(3):
        for k in range(len(fb)):
            u, v = uv[k, p]
            if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < d['W'] and 0 <= v < d['H']):
                assert k >= ns['N_MARKED'], 'marked point off screen'
                uv[k, p] = uv[k - 1, p]
                vis[k, p] = False
    ke = int(np.nonzero(fb == te)[0][0])
    # balance the answer letter over items: rotate letters so the closest at t_e gets letter (num % 3)
    closest = int(np.argmin(r[ke]))
    target = num % 3
    perm = [(k - target + closest) % 3 for k in range(3)]  # letter k <- original point perm[k]
    L3 = ['A', 'B', 'C']
    pts = {L3[k]: dict(uv=uv[:, perm[k]], r=r[:, perm[k]], vis=vis[:, perm[k]]) for k in range(3)}
    ans = L3[target]
    assert int(np.argmin([pts[L]['r'][ke] for L in L3])) == target
    data = dict(d, dataset='kitti_raw', video_id=c['drive'])
    q = ns['Q'].format(event=ns['EVENT_TEXT'][c['event']])
    extra = dict(t0=int(t0), te=int(te), tn=int(tn), event=c['event'], event_baked_frame=ke + 1,
                 closest_t0=L3[int(np.argmin([pts[L]['r'][0] for L in L3]))], selection=c, source_script=SCRIPT['T6'])
    write_item('T6', num, data, fb, pts, [te], q, L3, ans, extra)


def cmd_render(tpl, assign_path, nums):
    items = {it['num']: it['cand'] for it in json.load(open(assign_path))['items']}
    flag = '--drives' if tpl == 'T6' else '--scenes'
    ns, _ = load_ns(tpl, base_argv(tpl) + [flag, 'none'])
    if tpl == 'T3':
        ns['OUT'] = f'{WORK}/T3'
    done = existing(tpl)
    fails = 0
    for num in nums:
        if num in done or os.path.exists(f'{PROD}/{tpl}/{tpl}_{num:03d}.json'):
            print('skip existing', num, flush=True)
            continue
        c = items[num]
        t = time.time()
        try:
            if tpl == 'T6':
                render_kitti(ns, c, num)
            else:
                d = ns['load'](c['scene'])
                if tpl == 'T3':
                    d['thr'] = ns['OVL'] * d['W']
                fb, pts, lf, q, opts, ans, extra = render_po(tpl, ns, d, c, num)
                write_item(tpl, num, d, fb, pts, lf, q, opts, ans, extra)
                del d
        except Exception as e:
            fails += 1
            import traceback
            traceback.print_exc()
            print('RENDER FAIL', tpl, num, c.get('scene'), repr(e)[:300], flush=True)
        print(f'  {tpl}_{num:03d} took {time.time() - t:.0f} s', flush=True)
    print('RENDER DONE', tpl, nums, 'fails', fails, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'assign', 'render'])
    ap.add_argument('tpl', choices=['T1', 'T3', 'T4', 'T5', 'T6'])
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--per_job', type=int, default=5)
    ap.add_argument('--exclude_pending', action='store_true')
    ap.add_argument('--only_kind', default='')
    a = ap.parse_args()
    if a.cmd == 'select':
        cmd_select(a.tpl, a.rest[0])
    elif a.cmd == 'assign':
        cmd_assign(a.tpl, a.n, a.per_job, a.exclude_pending, a.only_kind)
    else:
        cmd_render(a.tpl, a.rest[0], [int(x) for x in a.rest[1].split(',')])
