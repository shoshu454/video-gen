# Stage 6: bake frames, draw markers, write input.mp4 / frames/ / overlay.mp4 / meta.json / qa.txt.
# Points are passed as per-baked-frame arrays so the same code serves track-based datasets
# (PointOdyssey, Dynamic Replica) and KITTI (LiDAR points propagated with OXTS).
import json, os
import cv2
import imageio_ffmpeg
import numpy as np

SHORT = 448
COLORS = {'A': (255, 220, 0), 'B': (0, 220, 255), 'C': (255, 60, 255)}  # RGB
N_MARKED = 3  # markers appear on the first 3 baked frames of input.mp4


def baked_size(H, W):
    s = SHORT / min(H, W)
    return int(round(W * s / 2)) * 2, int(round(H * s / 2)) * 2, s


def ring_radius(Wb):
    return max(4, int(round(0.015 * Wb)))


def draw_marker(img, u, v, letter, R, faded=False):
    col = COLORS[letter] if not faded else (150, 150, 150)
    c = (int(round(u)), int(round(v)))
    cv2.circle(img, c, R, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, c, R, col, 2, cv2.LINE_AA)
    org = (c[0] + R + 3, c[1] - R + 4)
    cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


def put(img, text, org, col=(255, 255, 255), scale=0.45):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    wr = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec='libx264', pix_fmt_out='yuv420p',
                                     macro_block_size=2, output_params=['-crf', '18'])
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f))
    wr.close()


def render_item(out_dir, data, frames, pts, label_frames, question, options, answer, meta_extra):
    """frames: baked source-frame indices (len 32). pts: {letter: {'uv': (32,2) px in source res,
    'r': (32,), 'vis': (32,) bool}}. label_frames: source indices the label depends on."""
    os.makedirs(f'{out_dir}/frames', exist_ok=True)
    H, W = data['H'], data['W']
    Wb, Hb, s = baked_size(H, W)
    R = ring_radius(Wb)
    fps = round((len(frames) - 1) / ((frames[-1] - frames[0]) / data['fps']), 3)  # real-time playback
    base = []
    for i in frames:
        base.append(cv2.resize(data['image'](i), (Wb, Hb), interpolation=cv2.INTER_AREA))

    inp = []
    for k, img in enumerate(base):
        img = img.copy()
        if k < N_MARKED:
            for L, p in pts.items():
                draw_marker(img, p['uv'][k, 0] * s, p['uv'][k, 1] * s, L, R)
        cv2.imwrite(f'{out_dir}/frames/{k:02d}.png', img[:, :, ::-1])
        inp.append(img)
    write_mp4(f'{out_dir}/input.mp4', inp, fps)

    ov = []
    lf = set(label_frames)
    for k, img in enumerate(base):
        img = img.copy()
        for L, p in pts.items():
            tr = (p['uv'][:k + 1] * s).astype(np.int32)
            if len(tr) > 1:
                cv2.polylines(img, [tr], False, COLORS[L], 1, cv2.LINE_AA)
        for L, p in pts.items():
            u, v = p['uv'][k] * s
            draw_marker(img, u, v, L, R, faded=not p['vis'][k])
            put(img, f"{p['r'][k]:.2f}m {'vis' if p['vis'][k] else 'OCC'}", (int(u) + R + 3, int(v) + R + 12), COLORS[L])
        t = (frames[k] - frames[0]) / data['fps']
        put(img, f'baked {k:02d}  src {frames[k]}  t={t:.2f}s', (8, 20), scale=0.55)
        if frames[k] in lf:
            cv2.rectangle(img, (0, 0), (Wb - 1, Hb - 1), (255, 40, 40), 6)
            put(img, 'LABEL FRAME', (8, 44), (255, 80, 80), 0.7)
        ov.append(img)
    write_mp4(f'{out_dir}/overlay.mp4', ov, fps)

    meta = dict(dataset=data['dataset'], video_id=data['video_id'], src_fps=data['fps'], video_fps=fps,
                src_resolution=[W, H], baked_resolution=[Wb, Hb], ring_radius_px=R,
                window=[int(frames[0]), int(frames[-1])], baked_frames=[int(i) for i in frames],
                label_frames=sorted(int(i) for i in lf), question=question, options=options, answer=answer,
                points={L: dict(uv_t0=[float(x) for x in p['uv'][0]],
                                r=[round(float(x), 4) for x in p['r']],
                                vis=[bool(x) for x in p['vis']]) for L, p in pts.items()})
    meta.update(meta_extra)
    json.dump(meta, open(f'{out_dir}/meta.json', 'w'), indent=1)
    with open(f'{out_dir}/qa.txt', 'w') as f:
        f.write(f'Q: {question}\n')
        if options:
            for o in options:
                f.write(f'{o}\n')
        f.write(f'A: {answer}\n')
