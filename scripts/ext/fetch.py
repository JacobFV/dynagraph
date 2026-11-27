"""EXT dataset fetcher (owner: agent EXT). Downloads ONLY no-registration, research-use-allowed files listed in
SOURCES to the LOCAL disk (/home/brandonin/datasets/<name>/raw/), resumable (curl -C -), then records sha256,
size, URL, license and fetch time in /home/brandonin/datasets/<name>/manifest.json and in the repo copy
results/ext/datasets_manifest.json. Never uploads anything. See docs/EXT.md for the license audit.

usage: python scripts/ext/fetch.py <group> [<group> ...]     (groups: keys of SOURCES)
       python scripts/ext/fetch.py --manifest-only            (re-hash what is on disk)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("EXT_DATA_ROOT", "/home/brandonin/datasets"))
REPO = Path(__file__).resolve().parents[2]
REPO_MANIFEST = REPO / "results" / "ext" / "datasets_manifest.json"
CAP_BYTES = 150e9   # PI cap on total download (raw bytes fetched)
FOOTPRINT_CAP = 300e9   # lead rule: total local footprint under ROOT (raw + extracted) <= 300 GB

S3P = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com"
GD = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="

# group -> (dataset name, license, license source, [(filename, url, expected bytes or None)])
SOURCES = {
    "physion_small": ("physion", "MIT (code + products; repo LICENSE)",
                      "https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/LICENSE", [
        ("Physion.zip", f"{S3P}/Physion.zip", 284049930),
        ("PhysionTrainMP4s.tar.gz", f"{S3P}/PhysionTrainMP4s.tar.gz", 803488088),
        ("model_human_results.zip", f"{S3P}/model_human_results.zip", 113211284),
        ("manual_stim_evaluation_glitchy_test_stims.txt", f"{S3P}/manual_stim_evaluation_glitchy_test_stims.txt", 1102),
    ]),
    "physionpp": ("physionpp", "MIT (NeurIPS 2023 D&B supplement: 'All products ... shared under the MIT license')",
                  "https://proceedings.nips.cc/paper_files/paper/2023/file/d3e8011c912e651ab2a76e7935a1e464-Supplemental-Datasets_and_Benchmarks.pdf", [
        ("human_data.zip", "https://physion-v2.s3.amazonaws.com/human_data.zip", 882404),
        ("readout_data.zip", "https://physion-v2.s3.amazonaws.com/readout_data.zip", 2971891321),
        ("test_data.zip", "https://physion-v2.s3.amazonaws.com/test_data.zip", 3892107787),
    ]),
    "dexycb_s01": ("dexycb", "CC BY-NC 4.0 (non-commercial; research use allowed)", "https://dex-ycb.github.io/", [
        ("calibration.tar.gz", GD + "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ", 14464),
        ("models.tar.gz", GD + "1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM", 1455399753),
        ("20200709-subject-01.tar.gz", GD + "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk", 12412314463),
    ]),
    "dexycb_s02_08": ("dexycb", "CC BY-NC 4.0 (non-commercial; research use allowed)", "https://dex-ycb.github.io/", [
        ("20200813-subject-02.tar.gz", GD + "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62", 12004145048),
        ("20200820-subject-03.tar.gz", GD + "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq", 12197037343),
        ("20200903-subject-04.tar.gz", GD + "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA", 12792618020),
        ("20200908-subject-05.tar.gz", GD + "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS", 12815420651),
        ("20200918-subject-06.tar.gz", GD + "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ", 12610980973),
        ("20200928-subject-07.tar.gz", GD + "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0", 12818333033),
        ("20201002-subject-08.tar.gz", GD + "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv", 12418522598),
    ]),   # lead decision 2026-11-26: H6 needs subjects 01-08 (TRAIN 01-05, VAL 06, TEST 07-08)
    "clevrer_val": ("clevrer", "CC0 (README.txt: 'Our dataset is under the CC0.')",
                    "http://data.csail.mit.edu/clevrer/README.txt", [
        ("README.txt", "http://data.csail.mit.edu/clevrer/README.txt", None),
        ("annotation_validation.zip", "http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip", None),
        ("video_validation.zip", "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip", None),
    ]),
    "mediapipe_models": ("mediapipe_models", "Apache-2.0 (MediaPipe model cards)",
                         "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker", [
        ("hand_landmarker.task", "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", None),
        ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", None),
    ]),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def total_on_disk() -> int:
    """bytes DOWNLOADED (files under */raw/, incl. partial prefixes/chunks); extracted copies are not downloads."""
    return sum(f.stat().st_size for f in ROOT.glob("*/raw/**/*") if f.is_file()) if ROOT.exists() else 0


def footprint() -> int:
    return sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file()) if ROOT.exists() else 0


def _deleted(name, fn) -> bool:
    m = ROOT / name / "manifest.json"
    return m.exists() and json.loads(m.read_text()).get(name, {}).get(fn, {}).get("deleted_after_extraction", False)


def record(name, lic, lic_src, fn, url, path):
    entry = dict(file=fn, url=url, bytes=path.stat().st_size, sha256=sha256(path), license=lic,
                 license_source=lic_src, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 local_path=str(path))
    for mpath in (ROOT / name / "manifest.json", REPO_MANIFEST):
        mpath.parent.mkdir(parents=True, exist_ok=True)
        m = json.loads(mpath.read_text()) if mpath.exists() else {}
        m.setdefault(name, {})[fn] = entry
        mpath.write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"[ok] {name}/{fn} {entry['bytes']/1e9:.3f} GB sha256={entry['sha256'][:16]}", flush=True)


def fetch(group: str):
    name, lic, lic_src, files = SOURCES[group]
    d = ROOT / name / "raw"
    d.mkdir(parents=True, exist_ok=True)
    for fn, url, nbytes in files:
        p = d / fn
        if _deleted(name, fn):
            print(f"[skip] {name}/{fn} already extracted and pruned", flush=True)
            continue
        if nbytes and not p.exists() and footprint() + 2 * nbytes > FOOTPRINT_CAP:   # archive + its extraction
            print(f"[cap] {name}/{fn}: footprint {footprint()/1e9:.0f} GB + 2x{nbytes/1e9:.1f} GB > 300 GB", flush=True)
            continue
        if nbytes and p.exists() and p.stat().st_size == nbytes:
            m = ROOT / name / "manifest.json"
            if not (m.exists() and fn in json.loads(m.read_text()).get(name, {})):
                record(name, lic, lic_src, fn, url, p)
            print(f"[skip] {name}/{fn} already complete", flush=True)
            continue
        elif nbytes and total_on_disk() + nbytes - (p.stat().st_size if p.exists() else 0) > CAP_BYTES:
            print(f"[cap] {name}/{fn} would exceed {CAP_BYTES/1e9:.0f} GB; skipped", flush=True)
            continue
        if "s3.amazonaws.com" in url and nbytes and nbytes > 1e8:     # S3 scales with connections
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from pget import pget
            pget(url, p, nbytes, nconn=8, log=lambda m: print(m, flush=True))
            record(name, lic, lic_src, fn, url, p)
            continue
        for attempt in range(20):
            r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
                                "-o", str(p), url])
            if r.returncode == 0 and (not nbytes or p.stat().st_size == nbytes):
                break
            if r.returncode == 33:   # server ignores ranges -> restart
                p.unlink(missing_ok=True)
            print(f"[retry {attempt}] {name}/{fn} rc={r.returncode}", flush=True)
            time.sleep(30)
        else:
            print(f"[FAIL] {name}/{fn}", flush=True)
            continue
        record(name, lic, lic_src, fn, url, p)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--manifest-only"]:
        for g, (name, lic, src, files) in SOURCES.items():
            for fn, url, _ in files:
                p = ROOT / name / "raw" / fn
                if p.exists():
                    record(name, lic, src, fn, url, p)
    else:
        for g in args:
            fetch(g)
        print(f"[done] total on disk {total_on_disk()/1e9:.2f} GB", flush=True)
