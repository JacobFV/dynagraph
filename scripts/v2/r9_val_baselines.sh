#!/usr/bin/env bash
# scripted VAL baselines on envs-r7.1 (13a): nominal, PTA, force-ramp (tight_insert), oracle-param; 30 VAL tasks per
# cell (make_tasks val, base 3_000_000 = the VAL demo tasks); 1 niced proc; resumable.
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh > /dev/null || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 EPISODE_TIMEOUT=600 OUT_DIR=results/v2_val_r9
for c in tight_insert:ur5e_2f85 tight_insert:panda fixture_place:ur5e_2f85 fixture_place:panda bin_clearing:ur5e_2f85 bin_clearing:panda; do
  f=${c%%:*}; r=${c##*:}; P="heuristic,heuristic_probe,oracle(labelled)"; [ $f = tight_insert ] && P="heuristic,heuristic_probe,force_ramp,oracle(labelled)"
  [ -f results/v2_val_r9/val_base_${f}_${r}.json ] || POLICIES=$P nice -n 10 python scripts/v2/r9_eval.py val_base_${f}_${r} $f $r val 30 3000000 >> logs/v2_r9_val_baselines.log 2>&1
done
echo R9_VAL_BASE_DONE >> logs/v2_r9_val_baselines.log
