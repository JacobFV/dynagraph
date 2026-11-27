#!/usr/bin/env bash
# V3-MODEL: push the v3 worktree to gb10-direct:~/dynagraph_v3/ (NOT ~/dynagraph; same excludes as scripts/sync.sh)
set -e
cd "$(dirname "$0")/../.."
rsync -az --exclude .git --exclude results --exclude ckpt --exclude data --exclude '__pycache__' --exclude '.venv' \
  ./ gb10-direct:~/dynagraph_v3/
