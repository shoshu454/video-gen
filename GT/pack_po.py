# Stream a PointOdyssey tar.gz once and store each scene's rgbs / depths / masks as one
# packed file <scene>/<kind>.blk + <kind>.idx.json ([name, offset, size, size, 0]) -- the
# shared NFS is far too slow for ~10k small files per scene. normals are skipped;
# anno.npz / info.npz / scene_info.json are written as plain files. Writes <scene>/.complete.
#   python pack_po.py <split_name> <archive or '-' for stdin>
import json, os, sys, tarfile, time

OUT = '/shared/rsaas/joeyw3/data/PointOdyssey/packed'
KINDS = {'rgbs', 'depths', 'masks'}
split, src = sys.argv[1], sys.argv[2]
fobj = sys.stdin.buffer if src == '-' else open(src, 'rb')
tf = tarfile.open(fileobj=fobj, mode='r|gz')

blk, idx, scenes = {}, {}, set()


def close_scene(scene):
    for k in [k for k in blk if k[0] == scene]:
        blk.pop(k).close()
        json.dump(idx.pop(k), open(f'{OUT}/{split}/{k[0]}/{k[1]}.idx.json', 'w'))
    open(f'{OUT}/{split}/{scene}/.complete', 'w').close()
    print(time.strftime('%H:%M:%S'), 'done', split, scene, flush=True)


cur = None
for m in tf:
    if not m.isfile():
        continue
    p = m.name.split('/')[1:]  # drop the top-level split dir: <split>/<scene>/<kind>/<file>
    if len(p) < 2:
        continue
    scene = p[0]
    if cur is not None and scene != cur and cur in scenes:
        close_scene(cur)
    cur = scene
    d = f'{OUT}/{split}/{scene}'
    if scene not in scenes:
        if os.path.exists(f'{d}/.complete'):
            print('skip existing', scene, flush=True)
        os.makedirs(d, exist_ok=True)
        scenes.add(scene)
    data = tf.extractfile(m).read()
    if len(p) == 3 and p[1] in KINDS:
        key = (scene, p[1])
        if key not in blk:
            blk[key] = open(f'{d}/{p[1]}.blk', 'wb')
            idx[key] = []
        off = blk[key].tell()
        blk[key].write(data)
        idx[key].append([p[2], off, len(data), len(data), 0])
    elif len(p) == 2:
        open(f'{d}/{p[1]}', 'wb').write(data)
if cur is not None:
    close_scene(cur)
print('ARCHIVE DONE', split, flush=True)
