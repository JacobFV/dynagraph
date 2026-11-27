#!/usr/bin/env bash
# MODEL-V2 round 6 (r9) TEST, pre-registered in docs/MODEL_V2.md 13a (run only after the 13a commit).
# n = 30 make_tasks(split=test / heldout_robot, base 3_000_000) per cell; skill-level policies (r9_eval.py) and the
# frozen ACT config (il_eval.py, real arms); <= 2 niced CPU procs; resumable; env asserted (envs-r7.1).
# usage: IL_RUN=<frozen> ENS_M=<frozen> scripts/v2/r9_test.sh
cd ~/dynagraph && source .venv/bin/activate
scripts/v2/m6_overlay.sh > /dev/null || exit 1
: "${IL_RUN:?}"; : "${ENS_M:?}"
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r7.1 \
       EPISODE_TIMEOUT=900 TAG=r9 TASKS=make_tasks IL_RUN ENS_M
cell() { f=${1%%:*}; rs=${1#*:}; r=${rs%%:*}; sp=${rs##*:}
  P="heuristic,heuristic_probe,oracle(labelled),typed_budget,generic_budget,nomem_budget,typed_full,generic_full,nomem_full"
  [ $f = tight_insert ] && P="heuristic,heuristic_probe,force_ramp,oracle(labelled),typed_budget,generic_budget,nomem_budget,typed_full,generic_full,nomem_full"
  [ -f results/v2/closed_loop/v2_r9_${f}_${r}.json ] || POLICIES=$P nice -n 10 python scripts/v2/r9_eval.py v2_r9_${f}_${r} $f $r $sp 30 3000000 >> logs/v2_r9_test_${f}_${r}.log 2>&1
  o=results/v2_il/test/il_${IL_RUN}_m${ENS_M}__${f}__${r}__test.json
  [ -f $o ] || nice -n 10 python scripts/v2/il_eval.py $IL_RUN $f $r $sp 30 3000000 $o >> logs/v2_r9_test_il_${f}_${r}.log 2>&1; }
export -f cell
mkdir -p results/v2_il/test
printf '%s\n' tight_insert:ur5e_2f85:test tight_insert:panda:test fixture_place:ur5e_2f85:test fixture_place:panda:test \
  bin_clearing:ur5e_2f85:test bin_clearing:panda:test bin_clearing:gen3_2f85:heldout_robot tight_insert:gen3_2f85:heldout_robot \
  | xargs -P ${PROCS:-2} -L 1 bash -c 'cell $0'
echo R9_TEST_DONE >> logs/v2_r9_test.log
