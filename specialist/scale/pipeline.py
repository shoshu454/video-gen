# CPU pipeline loop of an overnight scale-up job:  pipeline.py <tag> <templates> <n workers> [--fetch <list>]
# Every PERIOD_S: run_all.py on the templates (select the new clips, assign + render up to TARGET items per template,
# incremental eval B into <ROOT>/eval/B_<tag>.json + summary_<tag>.md). Ends after a final pass once every worker of
# the job has written its done marker, every template has TARGET items, or MAX_H hours have passed.
# --fetch: first start fetch_kitti_more.py on the list in the background (KITTI workers wait for <SCALE>/FETCH_DONE).
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE (vision-c22)
import glob, os, subprocess, sys, time

ROOT = '/shared/rsaas/joeyw3/pvideo_pseudo'
SCALE = f'{ROOT}/scale'
PSEUDO = '/home/joeyw3/world_model/pvideo/pseudo'
QWEN = '/shared/rsaas/joeyw3/miniconda3/envs/qwen_video/bin/python'
TARGET = 100
PERIOD_S = 1200
MAX_H = 40


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def counts(tpls):
    return {t: len(glob.glob(f'{ROOT}/qa/{t}/{t}_*.json')) for t in tpls}


def run_pass(tag, tpls):
    t = time.time()
    with open(f'{ROOT}/logs/scale/pipeline_{tag}.log', 'a') as f:
        f.write(f'\n==== pass {time.ctime()}\n')
        f.flush()
        rc = subprocess.run([QWEN, 'run_all.py', ','.join(tpls), f'--n_items={TARGET}', f'--eval_tag={tag}'],
                            cwd=PSEUDO, stdout=f, stderr=subprocess.STDOUT).returncode
    log('pass exit', rc, f'{time.time() - t:.0f} s', counts(tpls))


def main():
    tag, tpls, n_workers = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3])
    os.makedirs(f'{ROOT}/logs/scale', exist_ok=True)
    fetch = None
    if '--fetch' in sys.argv:
        lst = sys.argv[sys.argv.index('--fetch') + 1]
        log('starting KITTI fetch', lst)
        fetch = subprocess.Popen(f'{QWEN} fetch_kitti_more.py {lst} > {ROOT}/logs/scale/fetch_{tag}.log 2>&1 '
                                 f'; touch {SCALE}/FETCH_DONE', shell=True, cwd=PSEUDO)
    t_start = time.time()
    while True:
        workers_done = len(glob.glob(f'{SCALE}/workers/{tag}_*.done'))
        fetch_done = fetch is None or fetch.poll() is not None
        final = (workers_done >= n_workers and fetch_done) or all(n >= TARGET for n in counts(tpls).values()) \
            or time.time() - t_start > MAX_H * 3600
        log(f'workers done {workers_done}/{n_workers}, fetch done {fetch_done}, final pass {final}')
        run_pass(tag, tpls)
        if final:
            break
        time.sleep(PERIOD_S)
    log('pipeline done', counts(tpls))
    print(open(f'{ROOT}/eval/summary_{tag}.md').read() if os.path.exists(f'{ROOT}/eval/summary_{tag}.md') else '')


if __name__ == '__main__':
    main()
