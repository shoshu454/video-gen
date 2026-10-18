# Fetch KITTI raw pieces missing on disk, via HTTP range reads on the S3 raw zips.
#   python fetch_kitti.py oxts  <drive...>   -> oxts/ + calib + tracklets (small)
#   python fetch_kitti.py full  <drive...>   -> image_02 (all frames) + velodyne_points
# Output: /shared/rsaas/joeyw3/data/kitti_raw/<date>/<drive>_sync/...
import os
import sys
import time
import zipfile

sys.path.insert(0, '/shared/rsaas/joeyw3/kitti_depth_diff/scripts')
from httpzip import HttpFile

ROOT = '/shared/rsaas/joeyw3/data/kitti_raw'
S3 = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
mode, drives = sys.argv[1], sys.argv[2:]


def extract(zf, names, strip_to):
    names.sort(key=lambda n: zf.getinfo(n).header_offset)
    for n in names:
        dst = f'{strip_to}/{n}'
        if os.path.exists(dst) or n.endswith('/'):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst + '.tmp', 'wb').write(zf.read(n))
        os.replace(dst + '.tmp', dst)


for drive in drives:  # e.g. 2011_09_26_drive_0001
    date = drive[:10]
    t0 = time.time()
    try:
        if mode == 'oxts':
            if not os.path.exists(f'{ROOT}/{date}/calib_cam_to_cam.txt'):
                zf = zipfile.ZipFile(HttpFile(f'{S3}/{date}_calib.zip'))
                extract(zf, zf.namelist(), ROOT)
            zf = zipfile.ZipFile(HttpFile(f'{S3}/{drive}/{drive}_sync.zip'))
            extract(zf, [n for n in zf.namelist() if '/oxts/' in n], ROOT)
            try:
                zf = zipfile.ZipFile(HttpFile(f'{S3}/{drive}/{drive}_tracklets.zip'))
                extract(zf, zf.namelist(), ROOT)
            except Exception as e:
                print(drive, 'no tracklets', repr(e)[:100], flush=True)
        else:
            zf = zipfile.ZipFile(HttpFile(f'{S3}/{drive}/{drive}_sync.zip'))
            extract(zf, [n for n in zf.namelist() if '/image_02/' in n or '/velodyne_points/' in n], ROOT)
            open(f'{ROOT}/{date}/{drive}_sync/.complete', 'w').close()
        print(drive, mode, 'ok', f'{time.time()-t0:.0f}s', flush=True)
    except Exception as e:
        print(drive, mode, 'FAIL', repr(e)[:200], flush=True)
