# Define the evaluation clips: one clip per existing GT item (all 22 per template; the first N_FIRST per template,
# answers balanced, are the original v1 clips and keep their definitions), each the item's window padded by 2 s
# (KITTI: 3 s) on each side, plus N_EXTRA clips per template from the production GT candidates
# (<PROD>/_work/<T>/cands) that were not made into items: PO on scenes the template's items don't use, KITTI on
# windows of the item drives that don't overlap any other clip. Each clip is processed RGB-only by build.py; its
# GT points (item points, or the candidate's GT tracks at t0) are added as CoTracker probe queries, and the pseudo
# pipeline later selects its own items inside the clip. Writes <ROOT>/clips.json.
#   python make_clips.py [--n_extra 10] [--dry]
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from build import CLIPS, ROOT
from blk import PO, names

PROD = '/shared/rsaas/joeyw3/pvideo'
KR = '/shared/rsaas/joeyw3/data/kitti_raw'
N_FIRST = 10  # v1 clips (round robin over answers); their names, windows and probes must not change
PAD = dict(po=48, kitti=30)


def answer_key(m):
    if m['template'] == 'T4':
        return 'changed' if m['ranking_tn'] != m['ranking_t0'] else 'same'
    if m['template'] == 'T3':
        return m['kind']
    return m['answer']


def pick(tpl):
    """All GT items of the template: the v1 round-robin picks first, then the rest, each part in item order."""
    items = [json.load(open(f)) for f in sorted(glob.glob(f'{PROD}/{tpl}/{tpl}_*.json'))]
    groups = {}
    for m in items:
        groups.setdefault(answer_key(m), []).append(m)
    first = []
    while len(first) < N_FIRST and any(groups.values()):  # round robin over answers
        for k in sorted(groups):
            if groups[k] and len(first) < N_FIRST:
                first.append(groups[k].pop(0))
    ids = {m['item_id'] for m in first}
    return sorted(first, key=lambda m: m['item_id']) + [m for m in items if m['item_id'] not in ids]


def n_frames_po(sc):
    return len(np.load(f'{PO}/{sc}/anno.npz')['intrinsics'])


def n_frames_kitti(drv):
    return len(names(f'{KR}/{drv[:10]}/{drv}_sync', 'image_02'))


def item_clip(tpl, m):
    t0, tn = m['window']
    probes = lambda start: [dict(L=L, t=t0 - start, u=p['uv_t0'][0], v=p['uv_t0'][1]) for L, p in m['points'].items()]
    if tpl == 'T6':
        drv = m['video_id']
        start, end = max(0, t0 - PAD['kitti']), min(n_frames_kitti(drv) - 1, tn + PAD['kitti'])
        c = dict(name=m['item_id'], template=tpl, src='kitti', drive=drv, start=start, end=end,
                 drive_id=f"pgt{m['item_id']}_{drv}", gt_item=m['item_id'], gt_answer=m['answer'],
                 t0=t0 - start, tn=tn - start, te=m['te'] - start, event=m['event'], probes=probes(start))
        assert c['drive_id'][:10] == f"pgt{m['item_id']}_", c['drive_id']  # date dir = drive_id[:10], unique per clip
        return c
    sc = m['video_id']
    start, end = max(0, t0 - PAD['po']), min(n_frames_po(sc) - 1, tn + PAD['po'])
    return dict(name=m['item_id'], template=tpl, src='po', scene=sc, start=start, end=end, gt_item=m['item_id'],
                gt_answer=m['answer'], t0=t0 - start, tn=tn - start, probes=probes(start))


def balanced(cands):
    """Candidates in round-robin order over their GT label, best score first within a label."""
    groups = {}
    for c in sorted(cands, key=lambda c: -c['score']):
        groups.setdefault(c.get('label', c.get('kind')), []).append(c)
    out = []
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]:
                out.append(groups[k].pop(0))
    return out


def cand_clips(tpl, clips, n_extra, per_drive_max=3):
    """Extra clips from unused production GT candidates, labels balanced, one per scene (PO) / non-overlapping
    windows on the item drives, at most per_drive_max extras per drive (KITTI: candidates exist only there)."""
    cands = balanced([c for f in sorted(glob.glob(f'{PROD}/_work/{tpl}/cands/*.json')) for c in json.load(open(f))])
    out = []
    if tpl == 'T6':
        taken = {}  # drive -> [(start, end)] of clips on it
        for c in clips:
            if c['src'] == 'kitti':
                taken.setdefault(c['drive'], []).append((c['start'], c['end']))
        per_drive = {}
        for k in cands:
            if len(out) >= n_extra:
                break
            drv = k['drive']
            start, end = max(0, k['t0'] - PAD['kitti']), min(n_frames_kitti(drv) - 1, k['tn'] + PAD['kitti'])
            if per_drive.get(drv, 0) >= per_drive_max or any(start <= e and s <= end for s, e in taken.get(drv, [])):
                continue
            name = f'{tpl}_x{len(out) + 1:02d}'
            c = dict(name=name, template=tpl, src='kitti', drive=drv, start=start, end=end, drive_id=f'pgt{name}_{drv}',
                     gt_item=None, gt_answer=k['label'], t0=k['t0'] - start, tn=k['tn'] - start, te=k['te'] - start,
                     event=k['event'], probes=[])
            assert c['drive_id'][:10] == f'pgt{name}_', c['drive_id']
            taken.setdefault(drv, []).append((start, end))
            per_drive[drv] = per_drive.get(drv, 0) + 1
            out.append(c)
        return out
    used = {c['scene'] for c in clips if c['template'] == tpl}
    for k in cands:
        if len(out) >= n_extra:
            break
        sc = k['scene']
        if sc in used or not os.path.exists(f'{PO}/{sc}/.complete'):
            continue
        anno = np.load(f'{PO}/{sc}/anno.npz')
        start, end = max(0, k['t0'] - PAD['po']), min(len(anno['intrinsics']) - 1, k['tn'] + PAD['po'])
        uv = anno['trajs_2d'][k['t0'], np.atleast_1d(k['j'])]  # the candidate's GT points at t0 -> probes (T1: one j)
        name = f'{tpl}_x{len(out) + 1:02d}'
        out.append(dict(name=name, template=tpl, src='po', scene=sc, start=start, end=end, gt_item=None,
                        gt_answer=k.get('label', k.get('kind')), t0=k['t0'] - start, tn=k['tn'] - start,
                        probes=[dict(L='ABC'[i], t=k['t0'] - start, u=float(u), v=float(v)) for i, (u, v) in enumerate(uv)]))
        used.add(sc)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_extra', type=int, default=10)
    ap.add_argument('--dry', action='store_true', help='print, do not write clips.json')
    a = ap.parse_args()
    old = {c['name']: c for c in json.load(open(CLIPS))} if os.path.exists(CLIPS) else {}
    clips = []
    for tpl in ['T1', 'T3', 'T4', 'T5', 'T6']:
        new = [item_clip(tpl, m) for m in pick(tpl)]
        new += cand_clips(tpl, clips + new, a.n_extra)
        for c in new:
            if c['name'] in old:  # raw caches are keyed by clip name: an existing clip must keep its definition
                assert old[c['name']] == c, f"clip {c['name']} changed vs {CLIPS}"
            print(c['name'], c.get('scene', c.get('drive')), c['start'], c['end'], c['end'] - c['start'] + 1, 'frames',
                  c['gt_answer'], '' if c['gt_item'] else '(candidate)')
        print(f'{tpl}: {len(new)} clips', flush=True)
        clips += new
    if not a.dry:
        json.dump(clips, open(CLIPS, 'w'), indent=1)
    print(len(clips), 'clips', '(dry run)' if a.dry else f'-> {CLIPS}')
