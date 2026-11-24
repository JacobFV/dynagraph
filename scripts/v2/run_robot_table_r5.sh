#!/usr/bin/env bash
# MODEL-V2 r5: 3-seed paired real-robot table on site-dressed scenes (envs-r6), damage recorded per episode.
# remote side; <= PROCS CPU processes (CPU shared with ENVS: default 9); usage: run_robot_table_r5.sh <tag> <trust.json>
tag=${1:-r5}; trust=${2}; N=${N:-30}
C=results/v2/checkpoints
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 NO_PRIOR=1 DRESSING=site SEEDS=0,1,2 TAG=$tag TRUST=$trust NO_UNGATED=1
export RESIDUAL=$C/v2_robots_${tag}_typed_s0.pt,$C/v2_robots_${tag}_typed_s1.pt,$C/v2_robots_${tag}_typed_s2.pt
run() { ROBOT=$1 SPLIT=$2 python scripts/v2/env_closed_loop.py v2_tab5_${tag}_$1_$3 $3 $N > logs/v2_tab5_${tag}_$1_$3.log 2>&1; }
cells=()
FRAG="conveyor_sort bin_clearing singulation tool_return pouch_transfer twist_lock machine_tending tote_transfer"
OTHER="door_open drawer_open fixture_place"   # r6 priority: decision-relevant families; drawer_open = G5 held-out family
for r in ur5e_2f85 panda; do for f in $FRAG $OTHER; do cells+=("$r test $f"); done; done
for f in conveyor_sort bin_clearing door_open tool_return twist_lock fixture_place; do cells+=("gen3_2f85 heldout_robot $f"); done
cells+=("probe test depalletize" "ur5e_suction test depalletize" "fr3_suction test depalletize" "probe test conveyor_sort" "probe test singulation")
export -f run; export C N tag
printf '%s\n' "${cells[@]}" | xargs -P ${PROCS:-8} -L 1 bash -c 'run $0 $1 $2'
