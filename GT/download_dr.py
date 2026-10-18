# Selective Dynamic Replica downloader: for each train zip, fetch only the left-camera
# images / depths / trajectories / masks / instance_id_maps of every sequence using HTTP
# range requests (each (seq, kind) block is contiguous in the zip).
# Many small files are very slow on the shared NFS, so each (seq, kind) block is stored
# as one raw file <seq>/<kind>.blk (the zip's local-header+data bytes) plus <kind>.idx.json
# listing [name, data_offset, compress_size, file_size, compress_type]; every member's CRC
# is verified before the block is kept. Read members with dr_io.read_member.
# Writes <seq>/.complete when all kinds are stored; stops after --target complete
# sequences or when free disk drops below 100 GB.
#   python download_dr.py --target 200 [--start 1]
import argparse, json, os, shutil, struct, time, zlib
from concurrent.futures import ThreadPoolExecutor
import requests
from remotezip import RemoteZip

OUT = '/shared/rsaas/joeyw3/data/dynamic_replica/train_blk'
LINKS = '/shared/rsaas/joeyw3/data/dynamic_stereo/links.json'
KINDS = ['images', 'depths', 'trajectories', 'masks', 'instance_id_maps']
MIN_FREE = 100e9

ap = argparse.ArgumentParser()
ap.add_argument('--target', type=int, default=200)
ap.add_argument('--start', type=int, default=1)
ap.add_argument('--workers', type=int, default=6)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
urls = json.load(open(LINKS))['train']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def n_complete():
    return sum(os.path.exists(f'{OUT}/{s}/.complete') for s in os.listdir(OUT))


def fetch_block(url, members, dst):
    a = members[0].header_offset
    b = members[-1].header_offset + 30 + len(members[-1].filename.encode()) + 1024 + members[-1].compress_size
    for attempt in range(5):
        try:
            r = requests.get(url, headers={'Range': f'bytes={a}-{b}'}, stream=True, timeout=120)
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(1 << 22):
                buf += chunk
            idx = []
            for m in members:
                o = m.header_offset - a
                assert buf[o:o + 4] == b'PK\x03\x04', m.filename
                nlen, elen = struct.unpack('<HH', buf[o + 26:o + 30])
                d0 = o + 30 + nlen + elen
                data = bytes(buf[d0:d0 + m.compress_size])
                if m.compress_type == 8:
                    data = zlib.decompress(data, -15)
                assert len(data) == m.file_size and zlib.crc32(data) == m.CRC, m.filename
                idx.append([os.path.basename(m.filename), d0, m.compress_size, m.file_size, m.compress_type])
            with open(dst + '.blk.tmp', 'wb') as f:
                f.write(buf)
            os.replace(dst + '.blk.tmp', dst + '.blk')
            json.dump(idx, open(dst + '.idx.json', 'w'))
            return
        except Exception as e:
            log('retry', members[0].filename, repr(e)[:200])
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'failed block {members[0].filename}')


def fetch_seq(url, seq, groups):
    if os.path.exists(f'{OUT}/{seq}/.complete'):
        return
    os.makedirs(f'{OUT}/{seq}', exist_ok=True)
    for k in KINDS:
        if (seq, k) in groups and not os.path.exists(f'{OUT}/{seq}/{k}.idx.json'):
            fetch_block(url, groups[(seq, k)], f'{OUT}/{seq}/{k}')
    open(f'{OUT}/{seq}/.complete', 'w').close()
    log('done', seq)


for zi in range(args.start, len(urls)):
    if n_complete() >= args.target:
        break
    free = shutil.disk_usage(OUT).free
    if free < MIN_FREE:
        log('STOP low disk', free / 1e9)
        break
    url = urls[zi]
    log('zip', zi, url, 'complete so far', n_complete(), 'free GB', round(free / 1e9))
    z = RemoteZip(url)
    groups = {}
    for m in z.infolist():
        p = m.filename.split('/')
        if len(p) == 3 and p[0].endswith('_left') and p[1] in KINDS and p[2]:
            groups.setdefault((p[0], p[1]), []).append(m)
    for v in groups.values():
        v.sort(key=lambda m: m.header_offset)
    seqs = sorted({s for s, _ in groups})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda s: fetch_seq(url, s, groups), seqs))
log('FINISHED complete', n_complete())
