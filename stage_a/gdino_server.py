# Grounding DINO worker for stage_a.py. specialists.detect_boxes needs transformers with torch >= 2.4, so it runs in
# the qwen_video env as a long-lived child process: the model is loaded once per stage_a run, not once per video.
# Protocol (one JSON object per line):
#   stdin  {"frames": "<.npy (n, H, W, 3) uint8>", "vocab": ["person", "car", ...], "out": "<.json>"}
#   stdout "DONE <out>" or "ERROR <message>" once the result is written to <out>:
#          [{"types": [name, ...], "boxes": [[x0, y0, x1, y1], ...]}, ...] (one entry per frame)
# Everything else the libraries print goes to stderr.
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE, see specialist/build.py
import functools, json, os, sys, traceback

proto = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)  # library prints -> stderr
sys.stdout = sys.stderr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'specialist'))
import numpy as np
import transformers
import specialists as S

# detect_boxes calls from_pretrained on every call: cache the loaded processor / model for the whole run
for cls in (transformers.AutoProcessor, transformers.AutoModelForZeroShotObjectDetection):
    cls.from_pretrained = functools.lru_cache(maxsize=2)(cls.from_pretrained)

proto.write('READY\n')
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    try:
        vocab = [w.strip() for w in req['vocab'] if w.strip()]
        S.DET_TEXT = ' '.join(f'{w}.' for w in vocab)
        S.DET_TYPE = {w: w for w in sorted(vocab, key=len, reverse=True)}  # longest match first ("traffic light")
        det = S.detect_boxes(np.load(req['frames']))
        with open(req['out'] + '.tmp', 'w') as f:
            json.dump([dict(types=t, boxes=b.tolist()) for t, b in det], f)
        os.replace(req['out'] + '.tmp', req['out'])
        proto.write(f"DONE {req['out']}\n")
    except Exception:
        traceback.print_exc()
        proto.write('ERROR ' + traceback.format_exc().replace('\n', ' | ') + '\n')
