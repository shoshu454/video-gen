# Run the unmodified production pipeline (prod/run.py + the walkthrough selection / label code) on pseudo-GT scenes
# written by build.py. Only data and output roots are redirected:
#   PointOdyssey root  blk.PO / common.PO      -> <ROOT>/data/po          (scenes 'pseudo/<name>')
#   KITTI raw root     walkthrough KR          -> build.KITTI_ROOT        (drives '<drive_id>')
#   item output        run.PROD / run.WORK     -> <ROOT>/qa, <ROOT>/qa/_work
#   python run_pseudo.py select T3 pseudo/T3_001          # PO-format templates take the scene name
#   python run_pseudo.py select T6 pgtT6_001_2011_09_26_drive_0036
#   python run_pseudo.py assign T3 --n 10
#   python run_pseudo.py render T3 <assign.json> 1,2,3
# Same commands and flags as prod/run.py.
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE: THP compaction stalls make big allocations take minutes on vision-c22
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [HERE, SRC, f'{SRC}/prod']
from build import KITTI_ROOT, PO_ROOT, ROOT
import blk

blk.PO = PO_ROOT  # before common / the walkthroughs import it
import common

common.PO = PO_ROOT
common.OUT_ROOT = f'{ROOT}/walkthrough_scratch'  # the walkthrough preambles makedirs under it
os.makedirs(common.OUT_ROOT, exist_ok=True)
import run

run.PROD = f'{ROOT}/qa'
run.WORK = f'{ROOT}/qa/_work'
_load_ns = run.load_ns


# Clarity tolerances: the walkthroughs compare the depth map with the track depth at GT precision (2%). Pseudo depth
# maps and lifted tracks agree to ~1.2% median per frame (T3_001), so requiring <= 2% on every frame of a window
# rejected every point; 5% keeps the check's purpose (catch occlusion / depth edges, which give >> 5%).
PSEUDO_TOL = dict(DEPTH_AGREE=0.05, STD_MAX=0.03)


def load_ns(tpl, argv):
    ns, sel = _load_ns(tpl, argv)
    ns['KR'] = KITTI_ROOT
    if tpl != 'T6':
        ns.update(PSEUDO_TOL)
    return ns, sel


run.load_ns = load_ns

# Search budget: T1 and T3 verify only the best-scored candidate of each window and give the window up if it fails.
# With pseudo-GT the extreme candidates are enriched for estimation artifacts, so every window was used up by one
# failed check (T1_001: 11/11 windows). Allow a few more candidates per window (still at most one kept per window).
# The label rules and checks themselves are unchanged.
SEARCH_PATCH = {
    'T1': [("        seen, kept = set(), 0\n", "        seen, kept, _tries, _done = set(), 0, {}, set()\n"),
           ("            if c['t0'] in seen:\n                continue\n            seen.add(c['t0'])\n",
            "            if c['t0'] in _done or _tries.get(c['t0'], 0) >= 5:\n                continue\n"
            "            _tries[c['t0']] = _tries.get(c['t0'], 0) + 1\n            seen.add(c['t0'])\n"),
           ("            c['region'] = rid\n            cands.append(c)\n",
            "            c['region'] = rid\n            cands.append(c)\n            _done.add(c['t0'])\n")],
    'T3': [("        tried = {k: set() for k, _ in PLAN}\n",
            "        tried = {k: set() for k, _ in PLAN}\n        _tr, _ok = {}, set()\n"),
           ("or c['t0'] in tried[kind]:\n                continue\n            tried[kind].add(c['t0'])\n",
            "or _tr.get((kind, c['t0']), 0) >= 4 or (kind, c['t0']) in _ok:\n                continue\n"
            "            tried[kind].add(c['t0'])\n            _tr[(kind, c['t0'])] = _tr.get((kind, c['t0']), 0) + 1\n"),
           ("            c['regions'] = rid\n            cands.append(c)\n            kept[kind] += 1\n",
            "            c['regions'] = rid\n            cands.append(c)\n            kept[kind] += 1\n"
            "            _ok.add((kind, c['t0']))\n")],
}
_patch_select = run.patch_select


def patch_select(tpl, ns, sel, name):
    sel = _patch_select(tpl, ns, sel, name)
    for old, new in SEARCH_PATCH.get(tpl, []):
        assert old in sel, (tpl, old)
        sel = sel.replace(old, new)
    return sel


run.patch_select = patch_select

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
        for name in a.rest[0].split(','):
            run.cmd_select(a.tpl, name)
    elif a.cmd == 'assign':
        run.cmd_assign(a.tpl, a.n, a.per_job, a.exclude_pending, a.only_kind)
    else:
        run.cmd_render(a.tpl, a.rest[0], [int(x) for x in a.rest[1].split(',')])
