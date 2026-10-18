# Read members of the packed Dynamic Replica blocks written by download_dr.py.
import json, zlib
from functools import lru_cache

DR = '/shared/rsaas/joeyw3/data/dynamic_replica/train_blk'


@lru_cache(maxsize=64)
def index(seq, kind):
    return {r[0]: r[1:] for r in json.load(open(f'{DR}/{seq}/{kind}.idx.json'))}


def names(seq, kind):
    return sorted(index(seq, kind))


def read_member(seq, kind, name):
    off, csize, size, ctype = index(seq, kind)[name]
    with open(f'{DR}/{seq}/{kind}.blk', 'rb') as f:
        f.seek(off)
        data = f.read(csize)
    return zlib.decompress(data, -15) if ctype == 8 else data
