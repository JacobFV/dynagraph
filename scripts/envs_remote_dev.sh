#!/usr/bin/env bash
# like remote.sh but syncs to / runs in ~/dynagraph_dev (or $ENVS_DEV_DIR) so a generation running from ~/dynagraph
# (or from another dev dir) keeps its code
cd "$(dirname "$0")/.."
D="${ENVS_DEV_DIR:-dynagraph_dev}"
rsync -az --exclude .git --exclude results --exclude ckpt --exclude data --exclude '__pycache__' --exclude '.venv' \
  ./ gb10-direct:~/$D/
exec ssh -n gb10-direct "cd ~/$D && source ~/dynagraph/.venv/bin/activate && export PYTHONPATH=src MUJOCO_GL=egl && mkdir -p results/v2/envs logs && $*"
