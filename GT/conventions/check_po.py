# PointOdyssey convention check: extrinsics direction, projection of trajs_3d vs trajs_2d,
# depth PNG scale vs camera z, visibility vs depth z-buffer.
import io, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/joeyw3/world_model/pvideo')
from blk import PO, read

scene = sys.argv[1] if len(sys.argv) > 1 else 'sample/r4_new_f'
d = f'{PO}/{scene}'
a = np.load(f'{d}/anno.npz')
T3, T2, vis, val, K, E = a['trajs_3d'], a['trajs_2d'], a['visibs'], a['valids'], a['intrinsics'], a['extrinsics']
print(scene, 'frames', len(T3), 'points', T3.shape[1])
for fi in [0, len(T3) // 2, len(T3) - 1]:
    for name, M in [('E world->cam', E[fi]), ('inv(E) world->cam', np.linalg.inv(E[fi]))]:
        Xc = T3[fi] @ M[:3, :3].T + M[:3, 3]
        uv = Xc @ K[fi].T
        uv = uv[:, :2] / uv[:, 2:3]
        m = val[fi] & vis[fi]
        err = np.hypot(*(uv[m] - T2[fi][m]).T)
        print(f' f{fi} {name}: reproj median {np.median(err):.3f} p99 {np.percentile(err, 99):.3f}  z>0 frac {(Xc[m, 2] > 0).mean():.3f}')
    Xc = T3[fi] @ E[fi][:3, :3].T + E[fi][:3, 3]
    raw = np.array(Image.open(io.BytesIO(read(d, 'depths', f'depth_{fi:05d}.png')))).astype(np.float64)
    H, W = raw.shape
    m = val[fi] & vis[fi]
    u, v = np.round(T2[fi][m]).astype(int).T
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z = Xc[m][ok, 2]; r = raw[v[ok], u[ok]]
    scale = np.median(z / np.maximum(r, 1))
    print(f'  depth png {raw.dtype} {H}x{W} max {raw.max():.0f}; z/png median {scale:.6e} (1000/65535={1000/65535:.6e}); rel spread p90 {np.percentile(np.abs(r*scale-z)/z,90):.4f}')
    # is depth z or range? compare against range too
    rng = np.linalg.norm(Xc[m][ok], axis=1)
    print(f'  png*scale vs z rel err median {np.median(np.abs(r*scale - z)/z):.4f}; vs range {np.median(np.abs(r*scale - rng)/rng):.4f}')
    # visibility vs z-buffer on points in-bounds
    mv = val[fi]
    u, v = np.round(T2[fi][mv]).astype(int).T
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zz = (T3[fi][mv] @ E[fi][:3, :3].T + E[fi][:3, 3])[ok, 2]
    dd = raw[v[ok], u[ok]] * scale
    front = np.abs(zz - dd) / zz < 0.02
    vv = vis[fi][mv][ok]
    print(f'  vis agrees with z-buffer(2%): visible->front {front[vv].mean():.3f}, invisible->behind {(~front[~vv]).mean():.3f}')
print('K[0]', K[0].tolist())
print('K constant over time:', np.allclose(K, K[0]))
C = np.array([-e[:3, :3].T @ e[:3, 3] for e in E])
print('camera travel (m):', np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
