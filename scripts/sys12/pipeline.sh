#!/usr/bin/env bash
# INKLING System I/II full pipeline (resumable; each step skipped if its output exists). One niced process at a time.
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src MUJOCO_GL=egl PYTORCH_ALLOC_CONF=expandable_segments:True
G() { scripts/gpu_heavy.sh "$@"; }
wait_ok() { while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done
            until [ $(free -g | awk '/Mem:/ {print $7}') -ge ${1:-40} ]; do sleep 60; done; }
R=results/sys12; CK=$R/checkpoints
# 1. training (4 arms)
scripts/sys12/train_all.sh
# 2. gate test with the TRAINED gated adapters
[ -f $R/gate_test_trained.json ] || { wait_ok; G sys12_gate "nice -n 10 python scripts/sys12/gate_test.py --ckpt $CK/s1r1_inkling_gated_s0.pt --out $R/gate_test_trained.json" > logs/sys12/gate_trained.log 2>&1; }
# 3. language retention
[ -f $R/lang_eval.json ] || { wait_ok; G sys12_lang "nice -n 10 python scripts/sys12/lang_eval.py --gated $CK/s1r1_inkling_gated_s0.pt --ungated $CK/s1r1_inkling_ungated_s0.pt" > logs/sys12/lang_eval.log 2>&1; }
# 4. MODEL-V2 references on the same eval subsets: r7 typed s0 (2.05 M samples) and a sample-matched MODEL-V2
[ -f $R/eval/v2_r7_typed_s0.json ] || G sys12_evalv2 "nice -n 10 python -m physmem.sys12_train evalv2 --ckpt results/v2/checkpoints/v2_robots_r7_typed_s0.pt --tag v2_r7_typed_s0" > logs/sys12/evalv2.log 2>&1
[ -f $R/v2/checkpoints/v2_sys12_matched_typed_s0.pt ] || G sys12_v2matched "V2_RESULTS=$R/v2 nice -n 10 python -m physmem.v2_train train --data data/v2_r7_final --variant typed --run v2_sys12_matched_typed_s0 --seed 0 --steps 500 --bs 256 --eval-every 250 --robot-balance --no-test" > logs/sys12/v2matched.log 2>&1
[ -f $R/eval/v2_matched_typed_s0.json ] || G sys12_evalv2m "nice -n 10 python -m physmem.sys12_train evalv2 --ckpt $R/v2/checkpoints/v2_sys12_matched_typed_s0.pt --tag v2_matched_typed_s0" >> logs/sys12/evalv2.log 2>&1
# 5. pre-registered closed loop (docs/INKLING_SYS12.md section a)
export EPISODE_TIMEOUT=600
mkdir -p $R/closed_loop
for cell in bin_ur5e pouch_ur5e bin_gen3; do
  [ -f $R/closed_loop/s12r1_$cell.json ] && continue
  wait_ok
  { date -u +%FT%TZ; sha256sum src/physmem/envs/*.py src/physmem/skills/*.py src/physmem/embodiments/*.py src/physmem/models/v2/*.py scripts/sys12/closed_loop.py src/physmem/models/sys12.py; } > $R/closed_loop/provenance_$cell.txt 2>/dev/null
  G sys12_cl_$cell "CELL=$cell N=30 nice -n 10 python scripts/sys12/closed_loop.py s12r1" >> logs/sys12/cl_$cell.log 2>&1
done
echo SYS12_PIPELINE_DONE
