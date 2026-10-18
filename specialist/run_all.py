# After the specialist runs: build any missing KITTI layouts from the raw cache, run the unmodified pipeline
# (select -> assign -> render) on the ready pseudo clips, then evaluate against GT (eval B).
# PO templates (T1/T3/T4/T5) use every ready PO clip, whatever template it was made for; T6 uses the KITTI clips.
# select only runs on clips that have no candidate file yet, so the script can be re-run as more clips get ready.
# Templates run concurrently (select writes to a per-template scratch folder, so one template stays sequential).
#   python run_all.py [T1,T3,...] [--refuse] [--n_items=N] [--eval_tag=TAG]
#   (qwen_video env; fusion runs in the video env as a subprocess)
#   --n_items: items per template to reach (one per clip at most; default 10)
#   --eval_tag: eval B of these templates only, incremental, into <ROOT>/eval/B_<TAG>.json (default: all, B.json)
#   --refuse: first re-fuse every clip from the raw cache with the current build.py
import glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from build import CLIPS, KITTI_ROOT, PO_ROOT, RAW, ROOT

QWEN = '/shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python'
VIDEO = '/shared/rsaas/joeyw3/miniconda3/envs/video/bin/python'
N_ITEMS = 10


def sh(cmd):
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print('\n'.join(l for l in out.splitlines() if not l.startswith('  cand'))[-3000:], flush=True)
    return out


def ready(c):
    if c['src'] == 'kitti':
        return os.path.exists(f"{KITTI_ROOT}/{c['drive_id'][:10]}/{c['drive_id']}_sync/.blk_complete")
    return os.path.exists(f"{PO_ROOT}/pseudo/{c['name']}/.complete")


def refuse_all(clips, procs=6):
    """Re-run the (CPU) fusion of every clip from its raw cache with the current build.py, in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [c['name'] for c in clips if os.path.exists(f"{RAW}/{c['name']}.npz")]

    def one(n):
        r = subprocess.run([VIDEO, 'build.py', 'clip', n, '--fuse_only'], cwd=HERE, capture_output=True, text=True)
        return n, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]
    with ThreadPoolExecutor(procs) as ex:
        for n, rc, last in ex.map(one, todo):
            print('refuse', n, rc, last, flush=True)


def run_template(tpl, clips, n_items):
    if tpl == 'T6':
        cs = [c for c in clips if c['src'] == 'kitti' and ready(c)]
        names = [c['drive_id'] for c in cs]
    else:
        cs = [c for c in clips if c['src'] == 'po' and ready(c)]
        names = [f"pseudo/{c['name']}" for c in cs]
    todo = [n for n in names if not os.path.exists(f"{ROOT}/qa/_work/{tpl}/cands/{n.replace('/', '_')}.json")]
    print(f'==== {tpl}: {len(cs)} clips ready, {len(todo)} to select', flush=True)
    for i in range(0, len(todo), 20):  # in chunks, so a crash costs little and the log shows progress
        sh([QWEN, 'run_pseudo.py', 'select', tpl, ','.join(todo[i:i + 20])])
    done = len(glob.glob(f'{ROOT}/qa/{tpl}/{tpl}_*.json'))
    if done < n_items and cs:
        out = sh([QWEN, 'run_pseudo.py', 'assign', tpl, '--n', str(n_items - done), '--exclude_pending'])
        path = next((l.split()[1] for l in out.splitlines() if l.startswith('ASSIGN ')), None)
        if path:
            nums = [it['num'] for it in json.load(open(path))['items']]
            if nums:
                sh([QWEN, 'run_pseudo.py', 'render', tpl, path, ','.join(map(str, nums))])
    print(f'==== {tpl}: {len(glob.glob(f"{ROOT}/qa/{tpl}/{tpl}_*.json"))} items', flush=True)


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    N_ITEMS = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--n_items=')), N_ITEMS)
    tag = next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--eval_tag=')), '')
    tpls = args[0].split(',') if args else ['T1', 'T3', 'T4', 'T5', 'T6']
    clips = json.load(open(CLIPS))
    if '--refuse' in sys.argv:
        refuse_all([c for c in clips if c['template'] in tpls])
    if 'T6' in tpls:  # KITTI layouts live on local disk: fuse them here from the raw cache
        for c in clips:
            if c['src'] == 'kitti' and not ready(c) and os.path.exists(f"{RAW}/{c['name']}.npz"):
                sh([VIDEO, 'build.py', 'clip', c['name'], '--fuse_only'])
    with ThreadPoolExecutor(len(tpls)) as ex:
        list(ex.map(lambda t: run_template(t, clips, N_ITEMS), tpls))
    if tag:
        sh([QWEN, 'eval_gt.py', 'B', ','.join(tpls), tag])
    else:
        sh([QWEN, 'eval_gt.py', 'B'])  # eval B only (A is settled; C not needed)
