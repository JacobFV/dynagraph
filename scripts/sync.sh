#!/usr/bin/env bash
# push local source to gb10-direct (no delete; never touches remote results/ckpt/data)
set -e
cd "$(dirname "$0")/.."
rsync -az --exclude .git --exclude results --exclude ckpt --exclude data --exclude '__pycache__' --exclude '.venv' \
  ./ gb10-direct:~/dynagraph/
