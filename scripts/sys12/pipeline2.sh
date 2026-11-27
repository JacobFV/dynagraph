#!/usr/bin/env bash
# INKLING System I/II post-training pipeline, run from the FROZEN tree ~/dynagraph_sys12 (git HEAD export = envs-r6,
# + sys12 overlay; results/ckpt/data/logs symlinked to ~/dynagraph). Every closed-loop cell runs through
# scripts/sys12/cl_guard.py (ENV_VERSION == envs-r6 and envs/skills/embodiments hashes == ~/dynagraph_r6, else abort).
# Waits for the running train_all.sh (training arms) to finish first. Resumable / idempotent.
cd ~/dynagraph_sys12 && source ~/dynagraph/.venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_sys12/src MUJOCO_GL=egl PYTORCH_ALLOC_CONF=expandable_segments:True
G() { scripts/gpu_heavy.sh "$@"; }
wait_ok() { while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
            until [ $(free -g | awk '/Mem:/ {print $7}') -ge ${1:-40} ]; do sleep 60; done; }
while pgrep -f "[t]rain_all.sh|[s]ys12_train train" > /dev/null; do sleep 120; done
R=results/sys12; CK=$R/checkpoints
for arm in inkling_gated inkling_ungated inkling_random compact; do    # any arm the training queue did not finish
  run=s1r1_${arm}_s0; [ -f $CK/$run.pt ] && continue; wait_ok
  G sys12_$run "nice -n 10 python -m physmem.sys12_train train --arm $arm --run $run --steps 4000 --bs 32 --seed 0" >> logs/sys12/$run.log 2>&1
done
[ -f $R/gate_test_trained.json ] || { wait_ok; G sys12_gate "nice -n 10 python scripts/sys12/gate_test.py --ckpt $CK/s1r1_inkling_gated_s0.pt --out $R/gate_test_trained.json" > logs/sys12/gate_trained.log 2>&1; }
[ -f $R/lang_eval.json ] || { wait_ok; G sys12_lang "nice -n 10 python scripts/sys12/lang_eval.py --gated $CK/s1r1_inkling_gated_s0.pt --ungated $CK/s1r1_inkling_ungated_s0.pt" > logs/sys12/lang_eval.log 2>&1; }
[ -f $R/eval/v2_r7_typed_s0.json ] || G sys12_evalv2 "nice -n 10 python -m physmem.sys12_train evalv2 --ckpt results/v2/checkpoints/v2_robots_r7_typed_s0.pt --tag v2_r7_typed_s0" > logs/sys12/evalv2.log 2>&1
[ -f $R/v2/checkpoints/v2_sys12_matched_typed_s0.pt ] || G sys12_v2matched "V2_RESULTS=$R/v2 nice -n 10 python -m physmem.v2_train train --data data/v2_r7_final --variant typed --run v2_sys12_matched_typed_s0 --seed 0 --steps 500 --bs 256 --eval-every 250 --robot-balance --no-test" > logs/sys12/v2matched.log 2>&1
[ -f $R/eval/v2_matched_typed_s0.json ] || G sys12_evalv2m "nice -n 10 python -m physmem.sys12_train evalv2 --ckpt $R/v2/checkpoints/v2_sys12_matched_typed_s0.pt --tag v2_matched_typed_s0" >> logs/sys12/evalv2.log 2>&1
export EPISODE_TIMEOUT=600
mkdir -p $R/closed_loop
sha256sum $CK/s1r1_inkling_gated_s0.pt > $R/closed_loop/s1_ckpt_sha256.txt
for cell in bin_ur5e pouch_ur5e bin_gen3; do
  [ -f $R/closed_loop/s12r1_$cell.json ] && continue
  wait_ok
  G sys12_cl_$cell "CELL=$cell N=30 nice -n 10 python scripts/sys12/cl_guard.py s12r1" >> logs/sys12/cl_$cell.log 2>&1
  grep -q "cl_guard\] ABORT" logs/sys12/cl_$cell.log && { echo "GUARD ABORT in $cell"; break; }
done
echo SYS12_PIPELINE2_DONE
