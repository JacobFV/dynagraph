#!/usr/bin/env bash
# r9 VAL sanity of the memory-model planners (13a): trust gates from VAL AUC, then the gated residual planner with each
# model (typed / generic / nomem x budget / full) on 10 VAL tasks per cell (make_tasks val, base 3_000_000). 1 proc.
cd ~/dynagraph && source .venv/bin/activate
until grep -q R9_MEM_SEEDS_DONE logs/v2_r9_mem_seeds.log 2>/dev/null; do sleep 120; done
scripts/v2/m6_overlay.sh > /dev/null || exit 1
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 EPISODE_TIMEOUT=600 OUT_DIR=results/v2_val_r9 TAG=r9
python scripts/v2/r9_trust.py > logs/v2_r9_trust.log 2>&1
for c in tight_insert:ur5e_2f85 tight_insert:panda fixture_place:ur5e_2f85 fixture_place:panda bin_clearing:ur5e_2f85 bin_clearing:panda; do
  f=${c%%:*}; r=${c##*:}
  [ -f results/v2_val_r9/val_plan_${f}_${r}.json ] || POLICIES=typed_budget,generic_budget,nomem_budget,typed_full,generic_full,nomem_full nice -n 10 python scripts/v2/r9_eval.py val_plan_${f}_${r} $f $r val 10 3000000 >> logs/v2_r9_val_planners.log 2>&1
done
echo R9_VAL_PLAN_DONE >> logs/v2_r9_val_planners.log
