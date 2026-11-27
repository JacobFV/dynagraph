#!/usr/bin/env bash
# V3 §0e driver (run on gb10 from ~/dynagraph_v3 AFTER the §0e pre-registration commit). Waits for both VAL pilots,
# applies the pre-registered selection rule, trains seeds 1-2 of the chosen config, then evaluates on the FRESH
# test2 / test2_L4 seeds: chosen arm (3 seeds), frozen-matrix comparators, classical, oracle. One GPU job at a time.
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/dynagraph/.venv/bin/python
# VAL pilot of arm_i2_s0 once trained (VAL only)
until [ -f results/v3/checkpoints/arm_i2_s0.pt ]; do sleep 60; done
if [ ! -f results/v3/eval/arm_i2_s0_val_causal.json ]; then
  cat > logs/pilot_arm_i2_s0.sh <<EOS
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1
M="nice -n 19 $PY -m physmem.v3_chain_eval"
\$M predict --ckpt results/v3/checkpoints/arm_i2_s0.pt --split val --passes 4 && \$M causal --ckpt results/v3/checkpoints/arm_i2_s0.pt --split val --n 100
EOS
  nohup ~/dynagraph/scripts/v2/gpu_small.sh v3_pilot_arm_i2_s0 "bash ~/dynagraph_v3/logs/pilot_arm_i2_s0.sh" > logs/pilot_arm_i2_s0.log 2>&1 &
  echo $! > logs/pilot_arm_i2_s0.pid
  while kill -0 $(cat logs/pilot_arm_i2_s0.pid) 2>/dev/null; do sleep 30; done
fi
until [ -f results/v3/eval/arm_i_s0_val_causal.json ] && [ -f results/v3/eval/arm_i2_s0_val_causal.json ]; do sleep 60; done
ARM=$($PY scripts/v3/select_arm.py); echo "selected $ARM"
if [ "$ARM" = arm_i2 ]; then EXTRA="--rel-frac 0.15 --late-frac 0.2"; else EXTRA=""; fi
# CPU side (one niced process): simulator counterfactuals on the fresh seeds, then classical / oracle
( for args in "simcf --split test2 --n 200" "simcf --split test2_L4 --n 200" "classical --split test2" \
              "causal --ckpt classical --split test2 --n 200" "causal --ckpt oracle --split test2 --n 200" \
              "causal --ckpt classical --split test2_L4 --n 200" "causal --ckpt oracle --split test2_L4 --n 200"; do
    nice -n 19 $PY -m physmem.v3_chain_eval $args & echo $! > logs/arm_test2_cpu.child.pid; wait $!
  done; echo CPU2_DONE ) > logs/arm_test2_cpu.log 2>&1 &
for s in 1 2; do
  name=${ARM}_s$s
  [ -f results/v3/checkpoints/$name.pt ] && continue
  scripts/v3/gpu_job.sh $name -m physmem.v3_chain train --run $name --kind arm_i --cf-frac 0.3 $EXTRA --seed $s \
      --steps 4000 --bs 512 --log-every 200 --eval-every 1000
  while kill -0 $(cat logs/$name.pid) 2>/dev/null; do sleep 30; done
done
until grep -q CPU2_DONE logs/arm_test2_cpu.log 2>/dev/null || [ -f results/v3/eval/simcf_test2_L4.json ]; do sleep 60; done
for r in ${ARM}_s0 ${ARM}_s1 ${ARM}_s2 lbs_law_s0 lbs_law_s1 lbs_law_s2 v1_s0 v1_s1 v1_s2 lbs_nolaw_s0 lbs_nolaw_s1 lbs_nolaw_s2; do
  ck=results/v3/checkpoints/$r.pt
  until [ -f $ck ]; do sleep 120; done
  cat > logs/t2_$r.sh <<EOS
cd ~/dynagraph_v3
export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1
M="nice -n 19 $PY -m physmem.v3_chain_eval"
\$M predict --ckpt $ck --split test2 --passes 1,2,4,8 && \$M predict --ckpt $ck --split test2 --cond no_anchor && \
\$M predict --ckpt $ck --split test2_L4 && \$M causal --ckpt $ck --split test2 --n 200 && \
\$M causal --ckpt $ck --split test2_L4 --n 200
EOS
  nohup ~/dynagraph/scripts/v2/gpu_small.sh "v3_t2_$r" "bash ~/dynagraph_v3/logs/t2_$r.sh" > logs/t2_$r.log 2>&1 &
  echo $! > logs/t2_$r.pid
  while kill -0 $(cat logs/t2_$r.pid) 2>/dev/null; do sleep 30; done
done
echo ARM_TEST2_DONE
