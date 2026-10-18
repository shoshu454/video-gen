# Tiny stage_a test set: 2 videos with full GT, 2 with partial GT (PointOdyssey via adapters/pointodyssey.py) and
# 2 RGB-only videos (an .mp4 and a frame folder via adapters/generic.py).
#   python tests/make_test_set.py <test_root>    -> <test_root>/raw (generic adapter input), <test_root>/input
import ctypes
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
import os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'adapters'))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
import pointodyssey

root = os.path.abspath(sys.argv[1])
inp, raw = f'{root}/input', f'{root}/raw'
os.makedirs(inp, exist_ok=True)
os.makedirs(raw, exist_ok=True)

# full GT
pointodyssey.convert('sample/r4_new_f', inp, start=0, frames=72, max_tracks=2000, suffix='_full')
pointodyssey.convert('test/ani2', inp, start=0, frames=72, max_tracks=2000, suffix='_full')
# partial GT: depth + intrinsics (estimate masks, extrinsics scaled to GT depth, tracks)
pointodyssey.convert('test/ani2', inp, start=300, frames=90, drop={'masks', 'extrinsics', 'tracks'}, suffix='_depthK')
# partial GT: intrinsics + extrinsics + masks (estimate depth scaled to GT translation, tracks)
pointodyssey.convert('sample/r4_new_f', inp, start=200, frames=90, drop={'depth', 'tracks'}, suffix='_camMasks')

# RGB only: an mp4 (3 s) and a frame folder (2 s, 12 fps)
import av
import imageio.v3 as iio
src = '/home/joeyw3/bfaaf8ed-8aa7-5ab0-b779-6bfb48091734.mp4'
with av.open(src) as c:
    fps = float(c.streams.video[0].average_rate)
fr = [f for _, f in zip(range(int(3 * fps)), iio.imiter(src, plugin='pyav'))]
from fractions import Fraction
with av.open(f'{raw}/clip_mp4.mp4', 'w') as out:
    st = out.add_stream('libx264', rate=Fraction(fps).limit_denominator(1001))
    st.width, st.height, st.pix_fmt = fr[0].shape[1] // 2 * 2, fr[0].shape[0] // 2 * 2, 'yuv420p'
    for f in fr:
        out.mux(st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:st.height, :st.width, :3]), format='rgb24')))
    out.mux(st.encode())
os.makedirs(f'{raw}/clip_frames', exist_ok=True)
src2 = '/home/joeyw3/tabletennis1.mov'
with av.open(src2) as c:
    fps2 = float(c.streams.video[0].average_rate)
step = max(1, int(round(fps2 / 12)))
for k, f in enumerate(iio.imiter(src2, plugin='pyav')):
    if k >= 2 * fps2:
        break
    if k % step == 0:
        im = Image.fromarray(f[..., :3])
        if max(im.size) > 960:
            s = 960 / max(im.size)
            im = im.resize((int(im.size[0] * s) // 2 * 2, int(im.size[1] * s) // 2 * 2), Image.BILINEAR)
        im.save(f'{raw}/clip_frames/frame_{k:04d}.jpg', quality=95)
subprocess.run([sys.executable, f'{HERE}/adapters/generic.py', raw, inp, '--fps', '12'], check=True)
print('test set in', inp)
