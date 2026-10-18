# Standard format I/O shared by stage_a.py, the adapters and the checkers. Spec: README_FORMAT.md.
#   <dir>/<video>/ video.mp4 | rgb/00000.png..., meta.json, depth/, masks/ (+ names.json), tracks/, camera/
# Every writer is atomic: files go to a temp path and are renamed; directories (depth/, masks/, tracks/) are
# written as <name>.tmp-<pid>/ and renamed into place, so a modality directory that exists is complete.
import json, os, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

MODALITIES = ('intrinsics', 'masks', 'depth', 'extrinsics', 'tracks')
TRACK_FILES = ('tracks_2d.npy', 'visibility.npy', 'query_frame.npy', 'tracks_3d.npy')
DEPTH_MAX_M = 65.535  # uint16 millimetres


# ---------------------------------------------------------------- presence
def has_rgb(d):
    return os.path.exists(f'{d}/video.mp4') or os.path.isdir(f'{d}/rgb')


def present(d, mod):
    """Is modality `mod` present in video folder d (complete per the format's all-or-nothing rule)?"""
    if mod == 'intrinsics':
        return os.path.exists(f'{d}/camera/intrinsics.npy')
    if mod == 'extrinsics':
        return os.path.exists(f'{d}/camera/extrinsics.npy')
    if mod == 'depth':
        return os.path.isdir(f'{d}/depth')
    if mod == 'masks':
        return os.path.isdir(f'{d}/masks')
    if mod == 'tracks':
        return os.path.exists(f'{d}/tracks/tracks_2d.npy')
    raise ValueError(mod)


def video_dirs(root):
    """Sorted video folder names under root (folders that contain meta.json)."""
    return sorted(n for n in os.listdir(root) if os.path.exists(f'{root}/{n}/meta.json'))


# ---------------------------------------------------------------- readers
def read_meta(d):
    return json.load(open(f'{d}/meta.json'))


def frame_files(d, sub, ext='.png'):
    return sorted(f for f in os.listdir(f'{d}/{sub}') if f.endswith(ext) and f[:-len(ext)].isdigit())


def read_rgb(d, frames=None):
    """(T, H, W, 3) uint8. frames: optional list of indices to read (others are decoded but not kept)."""
    keep = None if frames is None else set(frames)
    out = []
    if os.path.exists(f'{d}/video.mp4'):
        import imageio.v3 as iio
        for i, f in enumerate(iio.imiter(f'{d}/video.mp4', plugin='pyav')):
            if keep is None or i in keep:
                out.append(f[..., :3])
            if keep is not None and i >= max(keep):
                break
    else:
        names = frame_files(d, 'rgb')
        idx = range(len(names)) if keep is None else sorted(keep)
        out = [np.array(Image.open(f'{d}/rgb/{names[i]}').convert('RGB')) for i in idx]
    return np.stack(out)


def count_rgb_frames(d):
    if os.path.exists(f'{d}/video.mp4'):
        import av
        with av.open(f'{d}/video.mp4') as c:
            return sum(1 for _ in c.decode(video=0))
    return len(frame_files(d, 'rgb'))


def _read_pngs(d, sub, frames=None):
    names = frame_files(d, sub)
    idx = range(len(names)) if frames is None else frames
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(lambda i: np.array(Image.open(f'{d}/{sub}/{names[i]}')), idx)))


def read_depth_mm(d, frames=None):
    """(T, H, W) uint16 millimetres, 0 = invalid."""
    return _read_pngs(d, 'depth', frames)


def read_depth(d, frames=None):
    """(T, H, W) float32 metres, 0 = invalid."""
    return read_depth_mm(d, frames).astype(np.float32) / 1000.0


def read_masks(d, frames=None):
    """(T, H, W) uint16 instance ids, 0 = background."""
    return _read_pngs(d, 'masks', frames)


def read_names(d):
    p = f'{d}/masks/names.json'
    return {int(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else None


def read_tracks(d):
    """dict uv (T, N, 2), vis (T, N), qt (N,) or None, X (T, N, 3) or None."""
    t = f'{d}/tracks'
    opt = lambda f: np.load(f'{t}/{f}') if os.path.exists(f'{t}/{f}') else None
    return dict(uv=np.load(f'{t}/tracks_2d.npy'), vis=np.load(f'{t}/visibility.npy'), qt=opt('query_frame.npy'),
                X=opt('tracks_3d.npy'))


def read_intrinsics(d):
    return np.load(f'{d}/camera/intrinsics.npy').astype(np.float64)


def read_extrinsics(d):
    """(T, 4, 4) camera-to-world."""
    return np.load(f'{d}/camera/extrinsics.npy').astype(np.float64)


# ---------------------------------------------------------------- atomic writers
def _tmp(path):
    return f'{path}.tmp-{os.getpid()}'


def write_json(path, obj):
    with open(_tmp(path), 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(_tmp(path), path)


def write_npy(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(_tmp(path), 'wb') as f:
        np.save(f, a)
    os.replace(_tmp(path), path)


def commit_dir(tmp, final):
    """Rename a finished temp directory into place (replacing a stale one)."""
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.rename(tmp, final)


def _write_pngs(dir_, arrays, threads=8):
    def one(i):
        Image.fromarray(arrays[i]).save(f'{dir_}/{i:05d}.png', compress_level=1)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(len(arrays))))


def depth_to_mm(z):
    """metres (float, nan/<=0 invalid) -> uint16 mm; beyond 65.535 m is stored as 0 (invalid)."""
    z = np.nan_to_num(np.asarray(z, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.round(z * 1000.0)
    mm[(mm <= 0) | (mm > 65535)] = 0
    return mm.astype(np.uint16)


def write_depth(d, depth_m):
    """depth_m: (T, H, W) metres -> <d>/depth/%05d.png (uint16 mm)."""
    tmp = _tmp(f'{d}/depth')
    os.makedirs(tmp, exist_ok=True)
    _write_pngs(tmp, [depth_to_mm(z) for z in depth_m])
    commit_dir(tmp, f'{d}/depth')


def write_masks(d, masks, names=None):
    """masks: (T, H, W) int instance ids (< 65536) -> <d>/masks/%05d.png (uint16) [+ names.json]."""
    assert masks.max() < 65536, masks.max()
    tmp = _tmp(f'{d}/masks')
    os.makedirs(tmp, exist_ok=True)
    _write_pngs(tmp, [m.astype(np.uint16) for m in masks])
    if names:
        with open(f'{tmp}/names.json', 'w') as f:
            json.dump({str(k): v for k, v in sorted(names.items())}, f, indent=1)
    commit_dir(tmp, f'{d}/masks')


def write_tracks(d, uv, vis, qt=None, X=None):
    tmp = _tmp(f'{d}/tracks')
    os.makedirs(tmp, exist_ok=True)
    np.save(f'{tmp}/tracks_2d.npy', np.asarray(uv, np.float32))
    np.save(f'{tmp}/visibility.npy', np.asarray(vis, bool))
    if qt is not None:
        np.save(f'{tmp}/query_frame.npy', np.asarray(qt, np.int64))
    if X is not None:
        np.save(f'{tmp}/tracks_3d.npy', np.asarray(X, np.float32))
    commit_dir(tmp, f'{d}/tracks')


def write_intrinsics(d, K):
    write_npy(f'{d}/camera/intrinsics.npy', np.asarray(K, np.float32))


def write_extrinsics(d, c2w):
    write_npy(f'{d}/camera/extrinsics.npy', np.asarray(c2w, np.float32))


def link_or_copy(src, dst):
    """Hard-link a file (copy across file systems)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_tree(src, dst):
    """Hard-link (or copy) a directory of files into place atomically."""
    tmp = _tmp(dst)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for f in sorted(os.listdir(src)):
        if os.path.isfile(f'{src}/{f}'):
            link_or_copy(f'{src}/{f}', f'{tmp}/{f}')
    commit_dir(tmp, dst)


# ---------------------------------------------------------------- validation (check_inputs / check_outputs)
def validate_video(d, require_all=False, deep=True):
    """Structural checks of one video folder. Returns (errors, warnings, info) lists of strings / a dict.
    deep: also read every depth / mask PNG (dtype, size); otherwise only the first and last."""
    err, warn, info = [], [], {}
    if not os.path.exists(f'{d}/meta.json'):
        return ['missing meta.json'], warn, info
    try:
        m = read_meta(d)
    except Exception as e:
        return [f'meta.json unreadable: {e}'], warn, info
    for k in ('dataset', 'source_path', 'fps', 'num_frames', 'resolution'):
        if k not in m:
            err.append(f'meta.json: missing "{k}"')
    if err:
        return err, warn, info
    T, (W, H) = int(m['num_frames']), m['resolution']
    if not (isinstance(m['fps'], (int, float)) and m['fps'] > 0):
        err.append(f'meta.json: fps must be a positive number, got {m["fps"]!r}')
    vid, rgbd = os.path.exists(f'{d}/video.mp4'), os.path.isdir(f'{d}/rgb')
    if vid == rgbd:
        err.append('need exactly one of video.mp4 and rgb/')
        return err, warn, info
    n = count_rgb_frames(d)
    if n != T:
        err.append(f'RGB has {n} frames, meta.num_frames = {T}')
    f0 = read_rgb(d, [0])[0]
    if f0.shape[:2] != (H, W):
        err.append(f'RGB frame is {f0.shape[1]}x{f0.shape[0]}, meta.resolution = {W}x{H}')
    if rgbd:
        names = frame_files(d, 'rgb')
        if names != [f'{i:05d}.png' for i in range(len(names))]:
            err.append('rgb/ files must be 00000.png, 00001.png, ... with no gaps')

    def check_pngs(sub, dtype_ok, what):
        names = frame_files(d, sub)
        if names != [f'{i:05d}.png' for i in range(len(names))]:
            err.append(f'{sub}/ files must be 00000.png, 00001.png, ... with no gaps')
        if len(names) != T:
            err.append(f'{sub}/ has {len(names)} frames, expected {T}')
            return
        idx = range(T) if deep else sorted({0, T - 1})
        for i in idx:
            a = np.array(Image.open(f'{d}/{sub}/{names[i]}'))
            if a.shape != (H, W):
                err.append(f'{sub}/{names[i]}: shape {a.shape}, expected ({H}, {W}) single channel')
                return
            if a.dtype not in dtype_ok:
                err.append(f'{sub}/{names[i]}: dtype {a.dtype}, expected 16-bit {what}')
                return

    if present(d, 'depth'):
        check_pngs('depth', (np.uint16, np.int32, np.uint32), 'millimetres')
        z = read_depth(d, sorted({0, T // 2, T - 1}))
        valid = z > 0
        info['depth_valid_frac'] = round(float(valid.mean()), 3)
        if valid.any():
            med = float(np.median(z[valid]))
            info['depth_median_m'] = round(med, 3)
            if med < 0.05 or med > 60:
                warn.append(f'depth median {med:.3f} m is implausible: are the units millimetres?')
        else:
            err.append('depth: no valid pixels in the sampled frames')
    if present(d, 'masks'):
        check_pngs('masks', (np.uint16, np.uint8, np.int32, np.uint32), 'instance ids')
        if os.path.exists(f'{d}/masks/names.json'):
            try:
                read_names(d)
            except Exception as e:
                err.append(f'masks/names.json: {e} (expected {{"<id>": "<name>"}})')
    if present(d, 'intrinsics'):
        K = np.load(f'{d}/camera/intrinsics.npy')
        if K.shape != (3, 3):
            err.append(f'intrinsics shape {K.shape}, expected (3, 3)')
        else:
            if not (K[0, 0] > 0 and K[1, 1] > 0 and abs(K[2, 2] - 1) < 1e-6 and np.allclose(K[2, :2], 0)
                    and abs(K[1, 0]) < 1e-9):
                err.append(f'intrinsics not of the form [[fx,s,cx],[0,fy,cy],[0,0,1]]: {K.tolist()}')
            if not (0 < K[0, 2] < W and 0 < K[1, 2] < H):
                warn.append(f'principal point ({K[0, 2]:.1f}, {K[1, 2]:.1f}) outside the {W}x{H} image')
            info['fx'] = round(float(K[0, 0]), 2)
            info['hfov_deg'] = round(float(np.degrees(2 * np.arctan(W / 2 / K[0, 0]))), 1)
            if K.dtype != np.float32:
                warn.append(f'intrinsics dtype {K.dtype}, expected float32')
    elif os.path.exists(f'{d}/camera') and not present(d, 'extrinsics'):
        warn.append('camera/ exists but holds neither intrinsics.npy nor extrinsics.npy')
    if present(d, 'extrinsics'):
        E = np.load(f'{d}/camera/extrinsics.npy')
        if E.shape != (T, 4, 4):
            err.append(f'extrinsics shape {E.shape}, expected ({T}, 4, 4)')
        else:
            R = E[:, :3, :3].astype(np.float64)
            orth = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
            det = np.linalg.det(R)
            if orth > 1e-3 or np.abs(det - 1).max() > 1e-3:
                err.append(f'extrinsics rotations are not proper rotations (max |RR^T - I| = {orth:.2e}, '
                           f'det in [{det.min():.3f}, {det.max():.3f}]); det = -1 means a flipped axis (OpenGL?)')
            if np.abs(E[:, 3] - [0, 0, 0, 1]).max() > 1e-6:
                err.append('extrinsics bottom row must be [0, 0, 0, 1]')
            if not np.isfinite(E).all():
                err.append('extrinsics contain nan / inf')
            C = E[:, :3, 3].astype(np.float64)
            info['camera_path_m'] = round(float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()), 3)
            if E.dtype != np.float32:
                warn.append(f'extrinsics dtype {E.dtype}, expected float32')
    if present(d, 'tracks'):
        t = f'{d}/tracks'
        uv = np.load(f'{t}/tracks_2d.npy', mmap_mode='r')
        if uv.ndim != 3 or uv.shape[0] != T or uv.shape[2] != 2:
            err.append(f'tracks_2d shape {uv.shape}, expected ({T}, N, 2)')
        else:
            N = uv.shape[1]
            info['n_tracks'] = N
            if uv.dtype != np.float32:
                err.append(f'tracks_2d dtype {uv.dtype}, expected float32')
            if not os.path.exists(f'{t}/visibility.npy'):
                err.append('tracks/visibility.npy missing (required with tracks_2d.npy)')
            else:
                v = np.load(f'{t}/visibility.npy', mmap_mode='r')
                if v.shape != (T, N) or v.dtype != bool:
                    err.append(f'visibility {v.shape} {v.dtype}, expected ({T}, {N}) bool')
                else:
                    vv = np.asarray(v)
                    u = np.asarray(uv)
                    inb = (u[..., 0] >= -0.5) & (u[..., 0] <= W - 0.5) & (u[..., 1] >= -0.5) & (u[..., 1] <= H - 0.5)
                    if vv.any():
                        out = float((vv & ~inb).sum() / vv.sum())
                        if out > 0.01:
                            warn.append(f'{out:.1%} of visible track points lie outside the image')
                        if not np.isfinite(u[vv]).all():
                            err.append('tracks_2d has nan / inf at visible points')
            if os.path.exists(f'{t}/query_frame.npy'):
                q = np.load(f'{t}/query_frame.npy')
                if q.shape != (N,) or not np.issubdtype(q.dtype, np.integer):
                    err.append(f'query_frame {q.shape} {q.dtype}, expected ({N},) int')
                elif len(q) and (q.min() < 0 or q.max() >= T):
                    err.append('query_frame values outside [0, T)')
            if os.path.exists(f'{t}/tracks_3d.npy'):
                X = np.load(f'{t}/tracks_3d.npy', mmap_mode='r')
                if X.shape != (T, N, 3) or X.dtype != np.float32:
                    err.append(f'tracks_3d {X.shape} {X.dtype}, expected ({T}, {N}, 3) float32')
    if require_all:
        for mod in MODALITIES:
            if not present(d, mod):
                err.append(f'missing modality: {mod}')
        if present(d, 'tracks'):
            for f in TRACK_FILES:
                if not os.path.exists(f'{d}/tracks/{f}'):
                    err.append(f'missing tracks/{f}')
    return err, warn, info


# ---------------------------------------------------------------- one-call writer for adapters
def write_rgb_frames(d, frames):
    """frames: (T, H, W, 3) uint8 array or iterable -> <d>/rgb/%05d.png."""
    tmp = _tmp(f'{d}/rgb')
    os.makedirs(tmp, exist_ok=True)
    _write_pngs(tmp, [np.asarray(f, np.uint8)[..., :3] for f in frames])
    commit_dir(tmp, f'{d}/rgb')


def write_video_folder(out_root, name, *, dataset, source_path, fps, rgb=None, rgb_mp4=None, depth_m=None,
                       masks=None, names=None, intrinsics=None, extrinsics_c2w=None, tracks_2d=None, visibility=None,
                       query_frame=None, tracks_3d=None, extra_meta=None, overwrite=False):
    """Write one video in the standard format. Give the RGB as `rgb` ((T, H, W, 3) uint8) or `rgb_mp4` (path of an
    existing .mp4, hard-linked / copied as is). Every other modality is optional (None = absent):
      depth_m (T, H, W) metres z-depth (nan / <= 0 invalid); masks (T, H, W) int ids, 0 = background;
      names {id: str}; intrinsics (3, 3); extrinsics_c2w (T, 4, 4) camera-to-world, OpenCV axes;
      tracks_2d (T, N, 2) xy px; visibility (T, N) bool; query_frame (N,) int; tracks_3d (T, N, 3) world metres.
    meta.json is written last, so a folder with meta.json is complete. Returns the folder path."""
    d = os.path.join(out_root, name)
    if os.path.exists(f'{d}/meta.json') and not overwrite:
        return d
    os.makedirs(d, exist_ok=True)
    if rgb_mp4 is not None:
        link_or_copy(rgb_mp4, f'{d}/video.mp4')
        import av
        with av.open(rgb_mp4) as c:
            s = c.streams.video[0]
            T = sum(1 for _ in c.decode(video=0))
            W, H = s.codec_context.width, s.codec_context.height
    else:
        rgb = np.asarray(rgb)
        assert rgb.ndim == 4 and rgb.shape[3] in (3, 4) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
        T, H, W = rgb.shape[:3]
        write_rgb_frames(d, rgb)

    def chk(a, shape, what):
        assert a.shape == shape, f'{what}: shape {a.shape}, expected {shape}'

    if depth_m is not None:
        chk(np.asarray(depth_m), (T, H, W), 'depth')
        write_depth(d, depth_m)
    if masks is not None:
        chk(np.asarray(masks), (T, H, W), 'masks')
        write_masks(d, np.asarray(masks), names)
    if intrinsics is not None:
        chk(np.asarray(intrinsics), (3, 3), 'intrinsics')
        write_intrinsics(d, intrinsics)
    if extrinsics_c2w is not None:
        chk(np.asarray(extrinsics_c2w), (T, 4, 4), 'extrinsics')
        write_extrinsics(d, extrinsics_c2w)
    if tracks_2d is not None:
        N = np.asarray(tracks_2d).shape[1]
        chk(np.asarray(tracks_2d), (T, N, 2), 'tracks_2d')
        assert visibility is not None, 'tracks_2d needs visibility'
        chk(np.asarray(visibility), (T, N), 'visibility')
        if query_frame is not None:
            chk(np.asarray(query_frame), (N,), 'query_frame')
        if tracks_3d is not None:
            chk(np.asarray(tracks_3d), (T, N, 3), 'tracks_3d')
        write_tracks(d, tracks_2d, visibility, query_frame, tracks_3d)
    meta = dict(dataset=dataset, source_path=str(source_path), fps=float(fps), num_frames=int(T), resolution=[int(W), int(H)])
    meta.update(extra_meta or {})
    write_json(f'{d}/meta.json', meta)
    return d
