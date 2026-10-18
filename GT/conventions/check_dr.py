# Dynamic Replica convention check: project traj_3d_world with the frame viewpoint and
# compare with traj_2d; compare camera z with traj_2d[:,2] and the depth map.
import gzip, io, json, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/joeyw3/world_model/pvideo')
from blk import DR, names as blk_names, read as blk_read


def names(seq, kind):
    return blk_names(f'{DR}/{seq}', kind)


def read_member(seq, kind, name):
    return blk_read(f'{DR}/{seq}', kind, name)

seq = sys.argv[1] if len(sys.argv) > 1 else '009850-3_obj_source_left'
ann = json.load(gzip.open('/shared/rsaas/joeyw3/data/dynamic_replica/frame_annotations_train.jgz'))
ann = {a['frame_number']: a for a in ann if a['image']['path'].startswith(seq + '/')}
ts = [ann[i]['frame_timestamp'] for i in sorted(ann)]
print('frames', len(ann), 'timestamps', ts[:4], 'fps~', 1 / np.median(np.diff(ts)))
for fi in [0, 100, 250]:
    a = ann[fi]
    vp = a['viewpoint']
    H, W = a['image']['size']
    s = min(H, W) / 2
    f = vp['focal_length'][0] * s
    cx, cy = W / 2 - vp['principal_point'][0] * s, H / 2 - vp['principal_point'][1] * s
    R, T = np.array(vp['R']), np.array(vp['T'])
    t = torch.load(io.BytesIO(read_member(seq, 'trajectories', f'{fi:06d}.pth')), weights_only=False)
    X = t['traj_3d_world'].numpy(); p2 = t['traj_2d'].numpy(); vis = t['verts_inds_vis'].numpy()
    Xc = X @ R + T  # pytorch3d row-vector convention: world -> camera
    u = cx - f * Xc[:, 0] / Xc[:, 2]; v = cy - f * Xc[:, 1] / Xc[:, 2]
    err = np.hypot(u - p2[:, 0], v - p2[:, 1])
    dn = [n for n in names(seq, 'depths') if n.endswith(f'_{fi:04d}.geometric.png')][0]
    d = np.frombuffer(np.array(Image.open(io.BytesIO(read_member(seq, 'depths', dn))), dtype=np.uint16), dtype=np.float16).astype(np.float32).reshape(H, W)
    ui, vi = np.round(p2[vis, 0]).astype(int), np.round(p2[vis, 1]).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    dz = d[vi[ok], ui[ok]]
    print(f'frame {fi}: f={f:.1f} cx={cx} cy={cy} H={H} W={W} npts={len(X)} vis={vis.mean():.2f}')
    print('  reproj err px: median %.3f p95 %.3f max %.3f' % (np.median(err), np.percentile(err, 95), err.max()))
    print('  |z_cam - traj_2d[:,2]| median %.4f; z range %.2f..%.2f' % (np.median(np.abs(Xc[:, 2] - p2[:, 2])), Xc[:, 2].min(), Xc[:, 2].max()))
    print('  depth map vs z_cam (visible, in-bounds): median rel err %.4f' % np.median(np.abs(dz - Xc[vis][ok, 2]) / Xc[vis][ok, 2]))
    print('  depth map range %.2f..%.2f' % (d[d > 0].min(), d.max()))
    print('  cam center world', -R @ T if False else -(T @ R.T))
