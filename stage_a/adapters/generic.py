# Generic adapter: any videos, no GT -> standard format with RGB + meta.json only.
#   python adapters/generic.py <src_dir> <out_dir> [--fps 30] [--dataset NAME]
# <src_dir> holds .mp4 / .mov / .mkv / .avi / .webm files and/or folders of frames (.png / .jpg, sorted by name).
#   .mp4 files are hard-linked as video.mp4 (other containers are re-encoded to PNG frames); fps is read from the
#   file. Frame folders become rgb/00000.png ...; they need --fps.
# Video names are the file / folder names without extension. Existing outputs (meta.json present) are skipped.
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import stdfmt as F

VIDEO_EXT = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')
FRAME_EXT = ('.png', '.jpg', '.jpeg')


def mp4_fps(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        r = s.average_rate or s.guessed_rate
        if not r:
            raise ValueError(f'{path}: no frame rate in the container')
        return float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--fps', type=float, default=0, help='frame rate of frame folders (required for them)')
    ap.add_argument('--dataset', default='generic')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for e in sorted(os.listdir(a.src_dir)):
        p = os.path.abspath(os.path.join(a.src_dir, e))
        name, ext = os.path.splitext(e)
        if os.path.isfile(p) and ext.lower() in VIDEO_EXT:
            fps = mp4_fps(p)
            if ext.lower() == '.mp4':
                F.write_video_folder(a.out_dir, name, dataset=a.dataset, source_path=p, fps=fps, rgb_mp4=p)
            else:
                import imageio.v3 as iio
                rgb = np.stack([f[..., :3] for f in iio.imiter(p, plugin='pyav')])
                F.write_video_folder(a.out_dir, name, dataset=a.dataset, source_path=p, fps=fps, rgb=rgb)
        elif os.path.isdir(p):
            files = sorted(f for f in os.listdir(p) if f.lower().endswith(FRAME_EXT))
            if not files:
                continue
            if not a.fps:
                sys.exit(f'{p} is a frame folder: pass --fps')
            rgb = np.stack([np.array(Image.open(os.path.join(p, f)).convert('RGB')) for f in files])
            F.write_video_folder(a.out_dir, e, dataset=a.dataset, source_path=p, fps=a.fps, rgb=rgb)
        else:
            continue
        n += 1
        print('wrote', os.path.join(a.out_dir, name if os.path.isfile(p) else e), flush=True)
    print(f'{n} videos', flush=True)


if __name__ == '__main__':
    main()
