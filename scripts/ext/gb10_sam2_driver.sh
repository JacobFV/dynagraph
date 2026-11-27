#!/usr/bin/env bash
# gb10 driver (owner: agent EXT): runs a SAM2 job list in chunks, ONE gpu_small.sh job at a time (each chunk waits
# for ~/dynagraph/.gpu_small.lock and >= 40 GB free RAM, as the shared-host rules require). Resumable: finished
# videos are skipped by sam2_tracks.py. usage: gb10_sam2_driver.sh <jobs.txt> <out_dir> <tag> [chunk=40]
set -u
jobs=$1; out=$2; tag=$3; n=${4:-40}
cd ~/dynagraph_ext || exit 1
mkdir -p chunks/$tag logs
split -l "$n" -d -a 3 "$jobs" chunks/$tag/c_
for c in chunks/$tag/c_*; do
  name="ext_sam2_${tag}_$(basename "$c")"
  nice -n 19 ~/dynagraph/scripts/v2/gpu_small.sh "$name" \
    "cd ~/dynagraph_ext && TQDM_DISABLE=1 PYTHONPATH=sam2_src:pylib:code/src ~/dynagraph/.venv/bin/python -P code/ext/sam2_tracks.py --list $c --out $out --ckpt ckpt/sam2.1_hiera_small.pt --stride 3 --max-obj 6" \
    >> logs/sam2_$tag.log 2>&1 < /dev/null
done
echo "[driver done] $tag" >> logs/sam2_$tag.log
