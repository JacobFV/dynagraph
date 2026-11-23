#!/usr/bin/env bash
# like remote.sh but syncs to / runs in ~/dynagraph_dev (so a generation running from ~/dynagraph keeps its code)
cd "$(dirname "$0")/.."
rsync -az --exclude .git --exclude results --exclude ckpt --exclude data --exclude '__pycache__' --exclude '.venv' \
  ./ gb10-direct:~/dynagraph_dev/
exec ssh -n gb10-direct "cd ~/dynagraph_dev && source ~/dynagraph/.venv/bin/activate && export PYTHONPATH=src MUJOCO_GL=egl && mkdir -p results/v2/envs logs && $*"
