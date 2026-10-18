# Fetch KITTI raw image_02 frames and Velodyne scans for a few drives, via HTTP range reads on the S3 _sync.zip,
# and store each kind as one packed <drive>_sync/<kind>.blk + <kind>.idx.json ([name, offset, size, size, 0],
# read with blk.read) -- the shared NFS is far too slow for thousands of small files.
# image_02: every frame of the drive (+ timestamps.txt). velodyne: only frames inside the given ranges.
#   python fetch_kitti_blk.py 2011_09_26_drive_0011:113-193 [drive:lo-hi[,lo-hi] ...]
# Writes <drive>_sync/.blk_complete when both kinds are stored.
import json, os, sys, time, zipfile
sys.path.insert(0, '/shared/rsaas/joeyw3/kitti_depth_diff/scripts')
from httpzip import HttpFile

ROOT = '/shared/rsaas/joeyw3/data/kitti_raw'
S3 = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'


def pack(zf, members, dst):
    members.sort(key=lambda m: m.header_offset)  # sequential reads hit the 8 MB range cache
    idx = []
    with open(dst + '.blk.tmp', 'wb') as f:
        for m in members:
            data = zf.read(m)  # zipfile checks the CRC
            idx.append([os.path.basename(m.filename), f.tell(), len(data), len(data), 0])
            f.write(data)
    os.replace(dst + '.blk.tmp', dst + '.blk')
    json.dump(idx, open(dst + '.idx.json', 'w'))


for arg in sys.argv[1:]:
    drive, rng = arg.split(':')
    ranges = [tuple(int(x) for x in r.split('-')) for r in rng.split(',')]
    d = f'{ROOT}/{drive[:10]}/{drive}_sync'
    if os.path.exists(f'{d}/.blk_complete'):
        print(drive, 'exists', flush=True)
        continue
    t0 = time.time()
    hf = HttpFile(f'{S3}/{drive}/{drive}_sync.zip')
    zf = zipfile.ZipFile(hf)
    info = zf.infolist()
    img = [m for m in info if '/image_02/data/' in m.filename and m.filename.endswith('.png')]
    ts = [m for m in info if m.filename.endswith('image_02/timestamps.txt')]
    vel = [m for m in info if '/velodyne_points/data/' in m.filename and m.filename.endswith('.bin')
           and any(lo <= int(os.path.basename(m.filename)[:10]) <= hi for lo, hi in ranges)]
    os.makedirs(d, exist_ok=True)
    if ts:
        open(f'{d}/image_02_timestamps.txt', 'wb').write(zf.read(ts[0]))
    pack(zf, img, f'{d}/image_02')
    pack(zf, vel, f'{d}/velodyne')
    open(f'{d}/.blk_complete', 'w').close()
    print(drive, f'{len(img)} images, {len(vel)} scans, {hf.n_bytes / 1e6:.0f} MB in {time.time() - t0:.0f} s', flush=True)
