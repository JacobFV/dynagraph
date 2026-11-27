"""Parallel HTTP range download (owner: agent EXT). S3 serves ~0.2 MB/s per connection to this host but scales with
connections, so large official files are fetched as fixed-size byte chunks over N connections and concatenated in
order. Resumable: finished chunks (exact size) are kept between runs.

pget(url, dest, end=None, start=0, nconn=8, chunk=32 MiB) appends bytes [len(dest) .. end) to dest
(dest may already hold a prefix, e.g. from an earlier run).
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def content_length(url) -> int:
    r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    return int(r.headers["Content-Length"])


def _chunk(url, a, b, path: Path):
    want = b - a
    for attempt in range(30):
        have = path.stat().st_size if path.exists() else 0
        if have == want:
            return
        if have > want:
            path.unlink()
            have = 0
        subprocess.run(["curl", "-sS", "--fail", "--max-time", "900", "-r", f"{a + have}-{b - 1}", url, "-o", "-"],
                       stdout=open(path, "ab"))
        if path.stat().st_size != want:
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"chunk {a}-{b} failed")


def pget(url, dest, end=None, nconn=8, chunk=32 << 20, log=print):
    dest = Path(dest)
    end = end or content_length(url)
    start = dest.stat().st_size if dest.exists() else 0
    if start >= end:
        return
    tmp = dest.parent / (dest.name + ".chunks")
    tmp.mkdir(exist_ok=True)
    spans = [(a, min(a + chunk, end)) for a in range(start, end, chunk)]
    t0 = time.time()
    with ThreadPoolExecutor(nconn) as ex:
        futs = [ex.submit(_chunk, url, a, b, tmp / f"{a:015d}") for a, b in spans]
        # append finished chunks in order as soon as the prefix is contiguous (keeps dest a valid prefix)
        with open(dest, "ab") as out:
            for i, ((a, b), f) in enumerate(zip(spans, futs)):
                f.result()
                p = tmp / f"{a:015d}"
                out.write(p.read_bytes())
                out.flush()
                p.unlink()
                if i % 8 == 0 or i == len(spans) - 1:
                    done = b - start
                    log(f"[pget] {dest.name} {b / 1e9:.2f}/{end / 1e9:.2f} GB "
                        f"{done / max(time.time() - t0, 1e-3) / 1e6:.2f} MB/s")
    tmp.rmdir()


if __name__ == "__main__":
    pget(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
