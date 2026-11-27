"""Fetch a PREFIX of each official Physion v1 readout-fitting archive (owner: agent EXT).

Why: the official readout sets are 17-45 GB per scenario (1000 stimuli each, frames stored inside HDF5) and this
host downloads at ~1.5 MB/s, so the full 1000 x 8 set (~250 GB) is out of budget. We download bytes [0, F*size) of
each official tar.gz with HTTP ranges (resumable), then extract every COMPLETE member of the gzip/tar prefix.
The result is the first ~F*1000 stimuli in archive order -- an unofficial reduced readout set (docs/EXT.md §3 labels
every number computed from it as "reduced readout, not the official protocol").

usage: python scripts/ext/physion_readout_prefix.py [--frac 0.2] [--scen Collide,...] [--extract-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import zlib
from pathlib import Path

S3 = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com"
ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets")) / "physion"
# official readout archives (the README's Roll link 'Rollreadout_HDF5s.tar.gz' is a 404; the bucket key is below)
SIZES = {"Dominoes": 20180963523, "Support": 42187707211, "Collide": 16935452948, "Contain": 39375766904,
         "Drop": 21121884859, "Link": 44538793512, "Drape": 40764087823, "Roll": 22399886098}


def url(s):
    return f"{S3}/{s}_readout_training_HDF5s.tar.gz"


def fetch_prefix(s, nbytes, part: Path, nconn=8):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pget import pget
    pget(url(s), part, nbytes, nconn=nconn, log=lambda m: print(f"[{s}] {m}", flush=True))


class _GzPrefix:
    """file-like over a truncated .gz that raises EOFError cleanly at the truncation point."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.buf = b""

    def read(self, n=-1):
        while n < 0 or len(self.buf) < n:
            c = self.f.read(1 << 20)
            if not c:
                break
            self.buf += self.d.decompress(c)
        out, self.buf = (self.buf, b"") if n < 0 else (self.buf[:n], self.buf[n:])
        return out


def extract(s, part: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    names = []
    tf = tarfile.open(fileobj=_GzPrefix(part), mode="r|")
    try:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".hdf5"):
                continue
            dst = out / Path(m.name).name
            fo = tf.extractfile(m)
            data = fo.read()
            if len(data) != m.size:          # truncated last member
                break
            if not (dst.exists() and dst.stat().st_size == m.size):
                dst.write_bytes(data)
            names.append(dst.name)
    except (EOFError, tarfile.ReadError, zlib.error):
        pass
    print(f"[{s}] {len(names)} complete stimuli extracted", flush=True)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--scen", default=",".join(SIZES))
    ap.add_argument("--extract-only", action="store_true")
    a = ap.parse_args()
    man_p = ROOT / "readout_prefix_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for s in a.scen.split(","):
        nb = int(SIZES[s] * a.frac)
        part = ROOT / "raw" / "readout_prefix" / f"{s}_readout_training_HDF5s.tar.gz.prefix"
        part.parent.mkdir(parents=True, exist_ok=True)
        if not a.extract_only:
            fetch_prefix(s, nb, part)
        names = extract(s, part, ROOT / "readout" / s)
        man[s] = dict(url=url(s), archive_bytes=SIZES[s], prefix_bytes=part.stat().st_size, frac=a.frac,
                      n_complete=len(names), stimuli=names,
                      note="prefix of official archive (archive order); unofficial reduced readout set")
        man_p.write_text(json.dumps(man, indent=1))
