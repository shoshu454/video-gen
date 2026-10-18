# Verify a stage_a.py output folder: every video with DONE has all modalities (incl. every tracks/ file) with correct
# shapes, dtypes and frame counts, and meta.json records the source of each modality.
#   python check_outputs.py <output_dir> [--quick]
# Also lists videos without DONE and the failures in failures.jsonl. Exit code 1 if a DONE video has errors.
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stdfmt as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir')
    ap.add_argument('--quick', action='store_true', help='only read the first / last depth and mask PNG')
    a = ap.parse_args()
    names = sorted(n for n in os.listdir(a.output_dir) if os.path.isdir(os.path.join(a.output_dir, n)))
    done = [n for n in names if os.path.exists(os.path.join(a.output_dir, n, 'DONE'))]
    bad = 0
    for n in done:
        d = os.path.join(a.output_dir, n)
        err, warn, info = F.validate_video(d, require_all=True, deep=not a.quick)
        if not err:
            m = F.read_meta(d)
            src = m.get('sources', {})
            for mod in F.MODALITIES:
                if mod not in src:
                    err.append(f'meta.json sources: no entry for {mod}')
            if 'scale_alignment' not in m:
                err.append('meta.json: no scale_alignment')
            if os.path.isdir(os.path.join(d, '_work')):
                warn.append('_work/ left behind')
            info['sources'] = ','.join(f'{k}:{v}' for k, v in src.items())
        bad += bool(err)
        print(f'[{"ERROR" if err else ("WARN" if warn else "OK")}] {n}  ' + '  '.join(f'{k}={v}' for k, v in info.items()))
        for e in err:
            print(f'    error: {e}')
        for w in warn:
            print(f'    warning: {w}')
    pending = [n for n in names if n not in done and os.path.exists(os.path.join(a.output_dir, n, 'meta.json'))
               or n not in done and os.path.isdir(os.path.join(a.output_dir, n, '_work'))]
    if pending:
        print(f'not finished ({len(pending)}): {", ".join(pending)}')
    fp = os.path.join(a.output_dir, 'failures.jsonl')
    if os.path.exists(fp):
        fails = [json.loads(l) for l in open(fp) if l.strip()]
        last = {}
        for f in fails:
            last[f['video']] = f
        print(f'failures.jsonl: {len(fails)} records; latest per video:')
        for v, f in sorted(last.items()):
            print(f'    {v} [{f["modality"]}] {f["error"]}' + ('  (since finished)' if v in done else ''))
    print(f'{len(done)} DONE, {bad} with errors')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
