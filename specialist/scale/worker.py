# GPU worker of the overnight scale-up jobs (one condor proc = one worker on 2 A40s):
#   worker.py <tag> <templates, e.g. T1,T3,T4> <work list file> <condor proc id>
# Takes clips from the work list in order and runs `build.py clip` on each (PO: specialists + fusion; KITTI:
# specialists only, the pipeline job fuses them on its local disk). The first GPU runs VDA / MonST3R / CoTracker, the
# second runs SAM2 (KITTI: the detector) at the same time (--parallel, PGT_CHILD_GPU). If that fails, the clip is
# retried serially on the first GPU, then given up. Clips are claimed with an atomic mkdir in <ROOT>/scale/claims,
# so workers of both jobs share clips without duplicates; a claim older than STALE_H without a result is taken over
# (its worker was evicted). Stops when every template of the job has TARGET items or nothing is left to build, and
# then writes <SCALE>/workers/<tag>_<proc>.done (the job's pipeline loop waits for all of them).
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE (vision-c22)
import glob, json, os, socket, subprocess, sys, time

ROOT = '/shared/rsaas/joeyw3/pvideo_pseudo'
SCALE = f'{ROOT}/scale'
CLAIMS = f'{SCALE}/claims'
LOGS = f'{ROOT}/logs/scale'
KR = '/shared/rsaas/joeyw3/data/kitti_raw'
PSEUDO = '/home/joeyw3/world_model/pvideo/pseudo'
VIDEO = '/shared/rsaas/joeyw3/miniconda3/envs/video/bin/python'
TARGET = 100
STALE_H = 3.0
WAIT_KITTI_S = 300  # KITTI clips whose drive is still being fetched: look again after this


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def n_items(tpl):
    return len(glob.glob(f'{ROOT}/qa/{tpl}/{tpl}_*.json'))


def gpu_free(g):
    """Free MiB of the GPU whose UUID starts with g (condor gives shortened UUIDs, which nvidia-smi -i rejects)."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        u, f = [x.strip() for x in line.split(',')]
        if u.startswith(g):
            return int(f)
    return 0


def built(c):
    if not os.path.exists(f"{ROOT}/raw/{c['name']}.npz"):
        return False
    return c['src'] == 'kitti' or os.path.exists(f"{ROOT}/data/po/pseudo/{c['name']}/.complete")


def kitti_ready(c):
    """All images of the clip window are packed (the fetch packs only clip windows of some drives)."""
    p = f"{KR}/{c['drive'][:10]}/{c['drive']}_sync/image_02.idx.json"
    if not os.path.exists(p):
        return False
    have = {int(r[0][:10]) for r in json.load(open(p))}
    return all(t in have for t in range(c['start'], c['end'] + 1))


def claim(name):
    d = f'{CLAIMS}/{name}'
    try:
        os.mkdir(d)
    except FileExistsError:
        if os.path.exists(f'{d}/FAILED') or time.time() - os.path.getmtime(d) < STALE_H * 3600:
            return False
        log('taking over stale claim', name, open(f'{d}/owner').read().strip() if os.path.exists(f'{d}/owner') else '')
        os.utime(d)
    open(f'{d}/owner', 'w').write(f'{socket.gethostname()} {os.getpid()} {time.ctime()}\n')
    return True


def build(c, gpus, parallel):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               PGT_SCRATCH=f'/tmp/pgt_scratch_{os.getpid()}', HF_HOME='/home/joeyw3/.cache/huggingface')
    env.pop('PGT_CHILD_GPU', None)
    args = [VIDEO, 'build.py', 'clip', c['name']]
    if c['src'] == 'kitti':
        args.append('--raw_only')
    if parallel:
        args.append('--parallel')
        env['PGT_CHILD_GPU'] = gpus[1]
    t = time.time()
    with open(f"{LOGS}/{c['name']}.log", 'a') as f:
        f.write(f"\n==== {socket.gethostname()} gpus {gpus} parallel={parallel} {time.ctime()}\n")
        f.flush()
        rc = subprocess.run(args, cwd=PSEUDO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    subprocess.run(['rm', '-rf', env['PGT_SCRATCH']])
    log(c['name'], 'parallel' if parallel else 'serial', 'exit', rc, f'{time.time() - t:.0f} s')
    return rc == 0


def main():
    tag, tpls, work, proc = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
    os.makedirs(CLAIMS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    gpus = [g for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g]
    log('worker', tag, socket.gethostname(), 'gpus', gpus)
    if not gpus:
        sys.exit('no GPU assigned')
    # Some condor GPUs are partly used by processes outside condor. Main GPU: VDA / MonST3R / CoTracker (~19 GiB +
    # fusion headroom); child GPU: SAM2 (~25 GiB). Use what is free; if neither GPU is, ask condor to retry (exit 99).
    free = {g: gpu_free(g) for g in gpus}
    log('free MiB', free)
    ok = sorted([g for g in gpus if free[g] >= 38000], key=lambda g: -free[g])
    if not ok:
        time.sleep(600)
        sys.exit(99)
    rest = [g for g in gpus if g != ok[0] and free[g] >= 28000]
    gpus = [ok[0]] + rest[:1]
    log('using', gpus, '(parallel)' if len(gpus) > 1 else '(serial)')
    clips = {c['name']: c for c in json.load(open(f'{ROOT}/clips.json'))}
    order = [l.strip() for l in open(work) if l.strip()]
    n_done = 0
    while True:
        if all(n_items(t) >= TARGET for t in tpls):
            log('target reached:', {t: n_items(t) for t in tpls})
            break
        pick, waiting = None, 0
        for name in order:
            c = clips[name]
            if built(c) or os.path.exists(f'{CLAIMS}/{name}/FAILED'):
                continue
            if c['src'] == 'kitti' and not kitti_ready(c):
                if not os.path.exists(f'{SCALE}/FETCH_DONE'):
                    waiting += 1
                continue
            if claim(name):
                pick = c
                break
        if pick is None:
            if waiting:
                log(f'{waiting} KITTI clips wait for their drive fetch; sleeping {WAIT_KITTI_S} s')
                time.sleep(WAIT_KITTI_S)
                continue
            log('nothing left to build')
            break
        ok = build(pick, gpus, parallel=len(gpus) > 1)
        if not ok and len(gpus) > 1:
            ok = build(pick, gpus, parallel=False)
        if not ok:
            open(f"{CLAIMS}/{pick['name']}/FAILED", 'w').write(time.ctime() + '\n')
        n_done += ok
    log('worker done,', n_done, 'clips built')
    os.makedirs(f'{SCALE}/workers', exist_ok=True)
    open(f'{SCALE}/workers/{tag}_{proc}.done', 'w').write(f'{socket.gethostname()} {n_done} clips {time.ctime()}\n')


if __name__ == '__main__':
    main()
