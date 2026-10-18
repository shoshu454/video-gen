# Read members of packed per-(sequence, kind) blocks: <seq_dir>/<kind>.blk + <kind>.idx.json,
# written by download_dr.py (Dynamic Replica) and pack_po.py (PointOdyssey).
import json, zlib
from functools import lru_cache

DR = '/shared/rsaas/joeyw3/data/dynamic_replica/train_blk'
PO = '/shared/rsaas/joeyw3/data/PointOdyssey/packed'


@lru_cache(maxsize=256)
def index(seq_dir, kind):
    return {r[0]: r[1:] for r in json.load(open(f'{seq_dir}/{kind}.idx.json'))}


def names(seq_dir, kind):
    return sorted(index(seq_dir, kind))


def read(seq_dir, kind, name):
    off, csize, size, ctype = index(seq_dir, kind)[name]
    with open(f'{seq_dir}/{kind}.blk', 'rb') as f:
        f.seek(off)
        data = f.read(csize)
    return zlib.decompress(data, -15) if ctype == 8 else data
