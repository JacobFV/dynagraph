"""Extract a fully downloaded official archive, verify it, then delete the archive (owner: agent EXT).

Lead rule (2026-11-26): the total local footprint under $EXT_DATA_ROOT (raw + extracted) stays <= 300 GB, and an
archive is deleted only after a verified extraction whose sha256 is already recorded in the manifest.
Steps: (1) re-hash the archive and require it to equal the manifest's sha256; (2) extract every member (tar/tar.gz
or zip; path-sanitised, no links); (3) verify every member exists with the right size; (4) delete the archive and mark
the manifest entry deleted_after_extraction (the URL + sha256 stay, so it can be re-fetched and checked).

usage: python -I scripts/ext/extract_and_prune.py <dataset> <file> <out_dir> [--keep]
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch  # noqa: E402


def _safe(out: Path, name: str) -> Path | None:
    d = (out / name).resolve()
    return d if str(d).startswith(str(out)) else None


def extract_verify(arc: Path, out: Path) -> int:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    if arc.suffix == ".zip":
        with zipfile.ZipFile(arc) as z:
            for i in z.infolist():
                if i.filename.startswith("__MACOSX/") or i.is_dir():
                    continue
                d = _safe(out, i.filename)
                if d is None:
                    continue
                if not (d.exists() and d.stat().st_size == i.file_size):
                    d.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(i) as src, open(d, "wb") as dst:
                        while b := src.read(1 << 22):
                            dst.write(b)
                assert d.stat().st_size == i.file_size, d
                n += 1
        return n
    with tarfile.open(arc, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            d = _safe(out, m.name)
            if d is None:
                continue
            if not (d.exists() and d.stat().st_size == m.size):
                d.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(d, "wb") as dst:
                    while b := src.read(1 << 22):
                        dst.write(b)
            assert d.stat().st_size == m.size, d
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("file")
    ap.add_argument("out")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    arc = fetch.ROOT / a.dataset / "raw" / a.file
    mpath = fetch.ROOT / a.dataset / "manifest.json"
    man = json.loads(mpath.read_text())
    ent = man[a.dataset][a.file]
    h = fetch.sha256(arc)
    if h != ent["sha256"]:
        sys.exit(f"sha256 mismatch for {arc}: {h} != manifest {ent['sha256']}")
    n = extract_verify(arc, Path(a.out))
    print(f"[verified] {a.file}: {n} members extracted to {a.out}", flush=True)
    if a.keep:
        return
    arc.unlink()
    for mp in (mpath, fetch.REPO_MANIFEST):
        m = json.loads(mp.read_text())
        e = m[a.dataset][a.file]
        e.update(deleted_after_extraction=True, extracted_to=str(Path(a.out).resolve()), n_members=n,
                 deleted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        mp.write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"[deleted] {arc}", flush=True)


if __name__ == "__main__":
    main()
