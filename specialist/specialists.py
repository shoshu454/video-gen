# Specialist models that stand in for the dataset GT. Run in the `video` conda env on a GPU.
#   metric depth z (m)           Video Depth Anything, metric ViT-L       -> PO depths, KITTI pseudo-LiDAR
#   camera K + poses + dyn mask  MonST3R global alignment (subsampled)    -> PO intrinsics/extrinsics, KITTI OXTS
#   2D tracks + visibility       CoTracker3 offline                       -> PO trajs_2d / visibs
#   instance masks               SAM2.1 auto masks + video propagation    -> PO masks
#   vehicle / person masks       Grounding DINO + SAM2.1 box prompts      -> KITTI tracklet boxes
# Every function takes frames as a (T, H, W, 3) uint8 array at the source resolution and returns arrays at that
# resolution. Models are loaded lazily and cached per process.
import math, os, shutil, sys, tempfile
import numpy as np
import torch
from PIL import Image

HOME = os.environ.get('PVIDEO_MODELS', '/home/joeyw3')  # folder holding the Video-Depth-Anything, monst3r, co-tracker clones
VDA = f'{HOME}/Video-Depth-Anything'
MONST3R = f'{HOME}/monst3r'
SAM2_REPO = f'{MONST3R}/third_party/sam2'  # stock SAM2 with the sam2.1 configs (the installed one is DAM4SAM's fork)
SAM2_CKPT = f'{SAM2_REPO}/checkpoints/sam2.1_hiera_large.pt'
COTRACKER_CKPT = f'{HOME}/co-tracker/checkpoints/scaled_online.pth'  # online beat offline on PO movers (9 vs 12 px)
SCRATCH = os.environ.get('PGT_SCRATCH', '/tmp/claude-1267708/pgt_scratch')  # local disk: small files are cheap here
_cache = {}
sys.path.insert(0, SAM2_REPO)  # before anything (MonST3R) imports the env's sam2 fork


def _scratch_dir(tag):
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.mkdtemp(prefix=f'{tag}_', dir=SCRATCH)


def _utils_pkg():
    """VDA and MonST3R's RAFT both import a top-level `utils` that is a namespace package (no __init__.py), so the
    unrelated `utils` package installed in the env would shadow them: register one `utils` spanning both dirs."""
    import types
    if getattr(sys.modules.get('utils'), '__pgt__', False):
        return
    pkg = types.ModuleType('utils')
    pkg.__path__ = [f'{VDA}/utils', f'{MONST3R}/third_party/RAFT/core/utils']
    pkg.__pgt__ = True
    sys.modules['utils'] = pkg


# ---------------------------------------------------------------- depth: Video Depth Anything (metric)
def run_vda(frames, fps):
    """(T, H, W) float32 metric z-depth in metres."""
    if 'vda' not in _cache:
        sys.path.insert(0, VDA)
        _utils_pkg()
        from video_depth_anything.video_depth import VideoDepthAnything
        sys.path.remove(VDA)
        for m in [m for m in sys.modules if m == 'utils' or m.startswith('utils.')]:
            del sys.modules[m]
        m = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], metric=True)
        m.load_state_dict(torch.load(f'{VDA}/checkpoints/metric_video_depth_anything_vitl.pth', map_location='cpu'),
                          strict=True)
        _cache['vda'] = m.cuda().eval()
    depths, _ = _cache['vda'].infer_video_depth(frames, fps, input_size=518, device='cuda', fp32=False)
    return np.asarray(depths, np.float32)


# ---------------------------------------------------------------- camera: MonST3R
def monst3r_crop(W1, H1, size=512):
    """MonST3R's crop_img geometry: resize the long side to `size`, centre-crop to multiples of 16.
    Returns (scale, x0, y0, w, h) such that crop pixel (x, y) = (orig * scale) - (x0, y0)."""
    S = max(W1, H1)
    W, H = [int(round(x * size / S)) for x in (W1, H1)]
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W == H:
        halfh = 3 * halfw / 4
    return W / W1, cx - halfw, cy - halfh, int(2 * halfw), int(2 * halfh)


def run_monst3r(frames, max_frames=80, niter=300):
    """Global alignment on <= max_frames evenly strided frames.
    Returns sel (frame indices used), cam2world (n, 4, 4), K (3, 3) in source pixels (shared focal), depth (n, h, w)
    and conf (n, h, w) in crop pixels, dyn (n, h, w) bool dynamic masks, crop (scale, x0, y0, w, h)."""
    T, H1, W1 = frames.shape[:3]
    stride = max(1, math.ceil(T / max_frames))
    sel = list(range(0, T, stride))
    if sel[-1] != T - 1:
        sel.append(T - 1)
    cwd = os.getcwd()
    os.chdir(MONST3R)  # the flow checkpoint paths inside MonST3R are relative to the repo
    if MONST3R not in sys.path:
        sys.path.insert(0, MONST3R)
    _utils_pkg()
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        if 'monst3r' not in _cache:
            _cache['monst3r'] = AsymmetricCroCo3DStereo.from_pretrained(f'{MONST3R}/checkpoints').cuda().eval()
        model = _cache['monst3r']
        tmp = _scratch_dir('monst3r')
        files = []
        for k, i in enumerate(sel):
            files.append(f'{tmp}/{k:05d}.png')
            Image.fromarray(frames[i]).save(files[-1])
        imgs = load_images(files, size=512, verbose=False)
        pairs = make_pairs(imgs, scene_graph='swinstride-5-noncyclic', prefilter=None, symmetrize=True)
        with torch.no_grad():
            output = inference(pairs, model, 'cuda', batch_size=16, verbose=False)
        scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False,
                               shared_focal=True, temporal_smoothing_weight=0.01, translation_weight=1.0,
                               flow_loss_weight=0.01, flow_loss_start_epoch=0.1, flow_loss_thre=25, use_self_mask=True,
                               num_total_iter=niter, empty_cache=len(sel) > 72, batchify=True)
        scene.compute_global_alignment(init='mst', niter=niter, schedule='linear', lr=0.01)
        with torch.no_grad():
            c2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
            Kc = scene.get_intrinsics().detach().cpu().numpy().astype(np.float64)
            depth = np.stack([d.detach().cpu().numpy() for d in scene.get_depthmaps()]).astype(np.float32)
            conf = np.stack([c.detach().cpu().numpy() for c in scene.get_conf()]).astype(np.float32)
            dm = scene.dynamic_masks if getattr(scene, 'sam2_dynamic_masks', None) is None else scene.sam2_dynamic_masks
            dyn = np.stack([np.asarray(m.detach().cpu().numpy() if torch.is_tensor(m) else m) for m in dm]).astype(bool)
        del scene, output, pairs
        torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        os.chdir(cwd)
    s, x0, y0, w, h = monst3r_crop(W1, H1)
    k = Kc[0]
    K = np.array([[k[0, 0] / s, 0, (k[0, 2] + x0) / s], [0, k[1, 1] / s, (k[1, 2] + y0) / s], [0, 0, 1]])
    return dict(sel=np.array(sel), cam2world=c2w, K=K, depth=depth, conf=conf, dyn=dyn, crop=np.array([s, x0, y0, w, h]))


# ---------------------------------------------------------------- tracks: CoTracker3 (online model, both directions)
def _cotracker():
    if 'cotracker' not in _cache:
        from cotracker.predictor import CoTrackerPredictor
        _cache['cotracker'] = CoTrackerPredictor(checkpoint=COTRACKER_CKPT, offline=False, window_len=16).cuda()
    return _cache['cotracker']


def track_queries(frames, queries, chunk=1500):
    """queries (N, 3) as (t, x, y) source pixels -> uv (T, N, 2), vis (T, N) bool."""
    model = _cotracker()
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
    uv, vis = [], []
    for i in range(0, len(queries), chunk):
        q = torch.from_numpy(np.asarray(queries[i:i + chunk], np.float32))[None].cuda()
        with torch.no_grad():
            tr, vi = model(video, queries=q, backward_tracking=True)
        uv.append(tr[0].cpu().numpy())
        vis.append(vi[0].cpu().numpy() > 0.5)
    del video
    torch.cuda.empty_cache()
    return np.concatenate(uv, 1).astype(np.float32), np.concatenate(vis, 1)


def run_cotracker(frames, fps, spacing=None, every_sec=1.0, extra=None):
    """Dense tracks seeded on a grid (spacing px) at frame 0 and every `every_sec` s; a later query frame only adds
    grid points with no visible track within 0.7 * spacing (fills newly revealed / entering surfaces).
    extra: optional (M, 3) probe queries tracked separately. Returns uv (T, N, 2), vis (T, N), qt (N,) query frame,
    and probe_uv / probe_vis."""
    T, H, W = frames.shape[:3]
    spacing = spacing or max(8, round(12 * max(H, W) / 960))  # 12 px at 960 wide: ~3.6k seeds per query frame
    xs = np.arange(spacing / 2, W, spacing)
    ys = np.arange(spacing / 2, H, spacing)
    gx, gy = [g.ravel() for g in np.meshgrid(xs, ys)]
    uv = np.zeros((T, 0, 2), np.float32)
    vis = np.zeros((T, 0), bool)
    qt = np.zeros(0, np.int32)
    from scipy.spatial import cKDTree
    for t in range(0, T, max(1, int(round(every_sec * fps)))):
        if uv.shape[1]:
            tree = cKDTree(uv[t][vis[t]]) if vis[t].any() else None
            far = np.ones(len(gx), bool) if tree is None else tree.query(np.stack([gx, gy], 1))[0] > 0.7 * spacing
        else:
            far = np.ones(len(gx), bool)
        if far.sum() < 10:
            continue
        q = np.stack([np.full(far.sum(), t), gx[far], gy[far]], 1)
        u, v = track_queries(frames, q)
        uv, vis, qt = np.concatenate([uv, u], 1), np.concatenate([vis, v], 1), np.r_[qt, np.full(len(q), t)]
        print(f'  cotracker: query frame {t}: +{len(q)} tracks ({uv.shape[1]} total)', flush=True)
    out = dict(uv=uv, vis=vis, qt=qt)
    if extra is not None and len(extra):
        out['probe_uv'], out['probe_vis'] = track_queries(frames, np.asarray(extra, np.float32))
    return out


# ---------------------------------------------------------------- instance masks: SAM2.1
def _sam2(kind):
    if SAM2_REPO not in sys.path:
        sys.path.insert(0, SAM2_REPO)
        for m in [m for m in sys.modules if m == 'sam2' or m.startswith('sam2.')]:
            del sys.modules[m]
    key = f'sam2_{kind}'
    if key not in _cache:
        import sam2
        assert sam2.__file__.startswith(SAM2_REPO), sam2.__file__
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_module
        GlobalHydra.instance().clear()  # another import may have initialised hydra with the other sam2 package
        initialize_config_module('sam2', version_base='1.2')
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        if kind == 'video':
            _cache[key] = build_sam2_video_predictor(cfg, SAM2_CKPT, device='cuda')
        else:
            _cache[key] = build_sam2(cfg, SAM2_CKPT, device='cuda', apply_postprocessing=False)
    return _cache[key]


def pick_instances(masks, H, W, min_area=0.002, bg_area=0.25):
    """Object-level, non-overlapping masks from SAM's many-granularity proposals. Larger first; a proposal is skipped
    when it is a group (>= 2 sub-proposals together cover >= 60% of it -- they are picked instead), when it is mostly
    inside an already chosen object-sized mask (a part), or a duplicate. A proposal inside a surface (> bg_area, e.g.
    the floor) is an object on it and is kept on top. Returns a list of bool masks."""
    ms = sorted([m for m in masks if m['area'] >= min_area * H * W], key=lambda m: -m['area'])
    segs = [m['segmentation'] for m in ms]
    area = np.array([s_.sum() for s_ in segs], float)
    chosen = []
    for i, seg in enumerate(segs):
        a = area[i]
        if a <= bg_area * H * W:
            kids = [j for j in range(i + 1, len(segs)) if (segs[j] & seg).sum() >= 0.8 * area[j]]
            if len(kids) >= 2 and (np.any([segs[j] for j in kids], 0) & seg).sum() >= 0.6 * a:
                continue
        if any((seg & c).sum() > 0.7 * a and c.sum() <= bg_area * H * W for c in chosen):
            continue
        if any((seg & c).sum() > 0.85 * c.sum() and (seg & c).sum() > 0.85 * a for c in chosen):  # duplicate
            continue
        chosen.append(seg)
    lab = np.zeros((H, W), np.int32)
    for k, c in enumerate(chosen):
        lab[c] = k + 1
    out = [lab == k + 1 for k in range(len(chosen))]
    return [o for o in out if o.sum() >= min_area * H * W]


def _label_from_logits(logits_by_obj, H, W):
    """{obj_id: (H, W) logits} -> (H, W) int label map (0 = background), per-pixel argmax over positive logits."""
    lab = np.zeros((H, W), np.int32)
    best = np.zeros((H, W), np.float32)
    for oid, lg in logits_by_obj.items():
        m = lg > best
        lab[m] = oid
        best[m] = lg[m]
    return lab


def duplicate_ids(lg, iou=0.5):
    """lg: per frame {obj_id: logits}. Two ids are one object seeded twice (at different key frames) when the median
    IoU of their positive masks, over sampled frames where both exist, is >= iou. Returns {dup_id: kept_id}."""
    T = len(lg)
    fr = range(0, T, max(1, T // 12))
    ids = sorted({o for f in fr for o in lg[f]})
    m = {o: [lg[f][o] > 0 if o in lg[f] else None for f in fr] for o in ids}
    remap = {}
    for i, a_ in enumerate(ids):
        for b_ in ids[i + 1:]:
            if b_ in remap or a_ in remap:
                continue
            ious = [(x & y).sum() / max((x | y).sum(), 1) for x, y in zip(m[a_], m[b_])
                    if x is not None and y is not None and x.any() and y.any()]
            if len(ious) >= 3 and np.median(ious) >= iou:
                remap[b_] = a_
    return remap


def run_sam2_instances(frames, fps, key_sec=2.0, points_per_side=32):
    """(T, H, W) uint8 instance label maps, temporally consistent ids, 0 = unlabeled (background).
    Pass 1: object masks from SAM2 automatic mask generation at frame 0, propagated through the clip.
    Pass 2: at every later key frame (each key_sec s), proposals covering < 30% labelled pixels are new objects. They
    are added as seeds together with every pass-1 object's mask on that key frame (SAM2 treats an object that has no
    prompt on a prompted frame as absent there), then everything is propagated forward and backward."""
    T, H, W = frames.shape[:3]
    _sam2('image')
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(_sam2('image'), points_per_side=points_per_side, pred_iou_thresh=0.7,
                                     stability_score_thresh=0.85, crop_n_layers=1,
                                     min_mask_region_area=int(0.001 * H * W))
    pred = _sam2('video')
    tmp = _scratch_dir('sam2')
    for i in range(T):
        Image.fromarray(frames[i]).save(f'{tmp}/{i:05d}.jpg', quality=95)
    keys = list(range(0, T, max(1, int(round(key_sec * fps)))))
    with torch.inference_mode():
        props = {k: pick_instances(gen.generate(frames[k]), H, W) for k in keys}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):

        def propagate(seeds):
            """seeds: [(obj_id, frame, mask)] -> per frame {obj_id: logits}"""
            state = pred.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
            for oid, f, m in seeds:
                pred.add_new_mask(state, frame_idx=f, obj_id=oid, mask=m)
            first = {}
            for oid, f, _ in seeds:
                first[oid] = min(first.get(oid, f), f)
            lg = [dict() for _ in range(T)]
            for rev in (False, True):
                for f, ids, logits in pred.propagate_in_video(state, reverse=rev):
                    for oid, l in zip(ids, logits[:, 0].half().cpu().numpy()):
                        # forward results from the object's first seed frame on, backward results before it
                        if (f >= first[oid]) != rev or f == first[oid]:
                            lg[f][oid] = l
            pred.reset_state(state)
            return lg

        seeds = [(k + 1, 0, m) for k, m in enumerate(props[0])]
        lg = propagate(seeds) if seeds else [dict() for _ in range(T)]
        lab = np.stack([_label_from_logits(l, H, W) for l in lg])
        nid = len(seeds)
        new = []
        for k in keys[1:]:
            for m in props[k]:
                if (lab[k][m] > 0).mean() < 0.3:
                    nid += 1
                    new.append((nid, k, m))
        if new:
            extra = []
            for k in sorted({f for _, f, _ in new}):  # re-prompt the existing objects on the new seeds' frames
                extra += [(oid, k, l > 0) for oid, l in lg[k].items() if oid <= len(seeds) and (l > 0).sum() >= 50]
            lg = propagate(seeds + extra + new)
        dups = duplicate_ids(lg)
        for dup, keep in dups.items():
            for l in lg:
                if dup in l:
                    x = l.pop(dup)
                    l[keep] = np.maximum(l[keep], x) if keep in l else x
        lab = np.stack([_label_from_logits(l, H, W) for l in lg])
        print(f'  sam2: {len(seeds)} seeds at frame 0, {len(new)} new objects at later key frames, {len(dups)} merged; '
              'labelled fraction per key frame', [round(float((lab[k] > 0).mean()), 2) for k in keys], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    u = np.unique(lab)  # renumber 1..n, 0 stays background
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[u[u > 0]] = np.arange(1, (u > 0).sum() + 1)
    lab = lut[lab]
    assert lab.max() < 256, lab.max()
    return lab.astype(np.uint8)


# ---------------------------------------------------------------- object masks for KITTI tracklets
DET_TEXT = 'car. van. truck. bus. person. bicycle. motorcycle. tram. train.'
DET_TYPE = {'car': 'Car', 'van': 'Van', 'truck': 'Truck', 'bus': 'Truck', 'person': 'Pedestrian', 'bicycle': 'Cyclist',
            'motorcycle': 'Cyclist', 'tram': 'Tram', 'train': 'Tram'}


def detect_boxes(frames, box_thr=0.35, text_thr=0.25):
    """Grounding DINO boxes per frame: list of (types, (n, 4) xyxy). Needs transformers with torch >= 2.4, so it runs
    in the qwen_video env (build.py detect)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    name = 'IDEA-Research/grounding-dino-base'
    proc, det = AutoProcessor.from_pretrained(name), AutoModelForZeroShotObjectDetection.from_pretrained(name).cuda().eval()
    out = []
    for f in frames:
        inp = proc(images=Image.fromarray(f), text=DET_TEXT, return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = det(**inp)
        r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                        target_sizes=[f.shape[:2]])[0]
        labels = r.get('text_labels', r.get('labels'))
        types, boxes = [], []
        for lab, b in zip(labels, r['boxes'].cpu().numpy()):
            word = next((w for w in DET_TYPE if w in str(lab)), None)
            if word is not None:
                types.append(DET_TYPE[word])
                boxes.append(b)
        out.append((types, np.array(boxes, np.float32).reshape(-1, 4)))
    return out


def run_detect_objects(frames, boxes):
    """Per frame: list of (type, bool mask) -- SAM2.1 masks from the detector boxes of detect_boxes()."""
    _sam2('image')
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sp = SAM2ImagePredictor(_sam2('image'))
    out = []
    for f, (types, bx) in zip(frames, boxes):
        items = []
        if len(bx):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                sp.set_image(f)
                ms, _, _ = sp.predict(box=bx, multimask_output=False)
            ms = np.asarray(ms).reshape(len(bx), -1, *f.shape[:2])[:, 0] > 0
            items = [(ty, m) for ty, m in zip(types, ms) if m.any()]
        out.append(items)
    return out
