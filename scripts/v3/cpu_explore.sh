#!/usr/bin/env bash
# V3 §0d exploratory CPU queue (ONE niced process at a time): waits for the frozen CPU eval to finish first.
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while kill -0 $(cat logs/eval_cpu.pid) 2>/dev/null; do sleep 30; done
for args in "physmem.v3_chain cfgen --split val" "physmem.v3_chain cfgen --split train" \
            "physmem.v3_chain gen --out data/v3_chain --split test2 --n 600" \
            "physmem.v3_chain gen --out data/v3_chain --split test2_L4 --n 300"; do
  nice -n 19 ~/dynagraph/.venv/bin/python -m $args &
  echo $! > logs/cpu_explore.child.pid; wait $! || echo "FAILED $args"
done
echo CPU_EXPLORE_DONE
