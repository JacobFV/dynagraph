#!/usr/bin/env bash
# V3 §0 TEST evaluation, CPU side (ONE niced process at a time): simulator counterfactuals, classical, oracle.
# Launched only after the pre-registration commit. Run on gb10 from ~/dynagraph_v3.
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPLIT=${SPLIT:-test}
for args in "simcf --split test --n 200" "simcf --split test_L4 --n 200" "classical --split test" \
            "causal --ckpt classical --split test --n 200" "causal --ckpt oracle --split test --n 200" \
            "classical --split test_L4" "causal --ckpt classical --split test_L4 --n 200" \
            "causal --ckpt oracle --split test_L4 --n 200"; do
  nice -n 19 ~/dynagraph/.venv/bin/python -m physmem.v3_chain_eval $args &
  echo $! > logs/eval_cpu.child.pid; wait $! || echo "FAILED $args"
done
echo EVAL_CPU_DONE
