#!/usr/bin/env bash
# MODEL-V2: paired real-robot table (n=30 per cell), heuristic / typed / residual(3-seed LCB) / generic / oracle.
# remote side; <=8 CPU processes; usage: scripts/v2/run_robot_table.sh <tag>
tag=${1:-t12b}; C=results/v2/checkpoints; N=${N:-30}
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 NO_PRIOR=1
export RESIDUAL=$C/v2_robots_${tag}_typed_s0.pt,$C/v2_robots_${tag}_typed_s1.pt,$C/v2_robots_${tag}_typed_s2.pt
run() { ROBOT=$1 SPLIT=$2 python scripts/v2/env_closed_loop.py v2_tab_${tag}_$1_$3 $3 $N $C/v2_robots_${tag}_typed_s0.pt $C/v2_robots_${tag}_generic_s0.pt > logs/v2_tab_${tag}_$1_$3.log 2>&1; }
cells=()
for r in ur5e_2f85 panda gen3_2f85; do for f in peg_insert drawer_open door_open twist_lock singulation fixture_place tool_return kitting_pocket; do
  sp=test; [ $r = gen3_2f85 ] && sp=heldout_robot; cells+=("$r $sp $f"); done; done
for f in depalletize; do cells+=("ur5e_suction test $f" "fr3_suction test $f"); done
for f in depalletize bulk_sweep conveyor_sort bin_clearing; do cells+=("probe test $f"); done
export -f run; export C N tag
printf '%s\n' "${cells[@]}" | xargs -P ${PROCS:-8} -L 1 bash -c 'run $0 $1 $2'
