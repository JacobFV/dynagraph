#!/usr/bin/env bash
# V3 §0 TEST evaluation, model side: one gpu_small job per checkpoint (sequential). Causal jobs wait for simcf.
cd ~/dynagraph_v3
RUNS=${RUNS:-"lbs_law_s0 lbs_law_s1 lbs_law_s2 v1_s0 v1_s1 v1_s2 lbs_nolaw_s0 lbs_nolaw_s1 lbs_nolaw_s2 dense_law_s0"}
until [ -f results/v3/eval/simcf_test.json ] && [ -f results/v3/eval/simcf_test_L4.json ]; do sleep 60; done
for r in $RUNS; do
  ck=results/v3/checkpoints/$r.pt
  [ -f $ck ] || { echo "missing $ck"; continue; }
  cat > logs/eval_$r.sh <<EOS
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M="nice -n 19 $HOME/dynagraph/.venv/bin/python -m physmem.v3_chain_eval"
\$M predict --ckpt $ck --split test --passes 1,2,4,8 && \$M predict --ckpt $ck --split test --cond no_anchor && \
\$M predict --ckpt $ck --split test --cond no_track && \$M predict --ckpt $ck --split test --cond no_proprio && \
\$M predict --ckpt $ck --split test_L4 && \$M causal --ckpt $ck --split test --n 200 && \
\$M causal --ckpt $ck --split test_L4 --n 200
EOS
  nohup ~/dynagraph/scripts/v2/gpu_small.sh "v3_eval_$r" "bash ~/dynagraph_v3/logs/eval_$r.sh" > logs/eval_$r.log 2>&1 &
  echo $! > logs/eval_$r.pid
  while kill -0 $(cat logs/eval_$r.pid) 2>/dev/null; do sleep 30; done
  echo "[$(date -u +%T)] eval $r rc=$(grep -o 'END v3_eval_'$r' rc=[0-9]*' ~/dynagraph/logs/gpu_small_ledger.log | tail -1)"
done
echo EVAL_GPU_DONE
