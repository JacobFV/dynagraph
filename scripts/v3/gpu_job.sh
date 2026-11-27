#!/usr/bin/env bash
# V3-MODEL GPU job on gb10-direct through the shared small-GPU lane (~/dynagraph/scripts/v2/gpu_small.sh:
# .gpu_small.lock, one job at a time, waits for >= 40 GB available RAM, jobs < 6 GB). Runs from ~/dynagraph_v3.
# usage (on gb10): scripts/v3/gpu_job.sh <name> <python -m module args...>
# Records the wrapper PID in logs/<name>.pid; the job's own python PID is written by the job to logs/<name>.child.pid
name=$1; shift
cd ~/dynagraph_v3
mkdir -p logs
cmd="cd ~/dynagraph_v3 && export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && nice -n 19 ~/dynagraph/.venv/bin/python $* & echo \$! > ~/dynagraph_v3/logs/$name.child.pid; wait \$!"
nohup ~/dynagraph/scripts/v2/gpu_small.sh "v3_$name" "$cmd" > logs/$name.log 2>&1 &
echo $! > logs/$name.pid
echo "launched $name pid $(cat logs/$name.pid)"
