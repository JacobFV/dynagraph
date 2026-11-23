#!/usr/bin/env bash
# run a command on gb10-direct inside the project venv: scripts/remote.sh "python -m physmem.cli doctor"
cd "$(dirname "$0")/.." && ./scripts/sync.sh
exec ssh -n gb10-direct "cd ~/dynagraph && source .venv/bin/activate && export PYTHONPATH=src MUJOCO_GL=egl && $*"
