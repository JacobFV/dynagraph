#!/usr/bin/env bash
# MODEL-V2 round 3 RESUME (remote) after the shared-host hold. Idempotent: every step skips existing outputs and waits
# while ~/dynagraph/HOLD exists. Order: (1) r6i1 typed s2 -> (2) trust r6i1 -> (3) round-2 rollouts -> (4) final stage
# (tranche8 conversion, final 3-seed models, trust, robot table) -> (5) paired videos for the best cells.
set -x
cd ~/dynagraph
hold() { while [ -f ~/dynagraph/HOLD ]; do sleep 60; done; }
export EXCLUDE_TRAINVAL_FAMILY=drawer_open
# (1) the interrupted interim run (killed at ~60%, no partial checkpoint)
if [ ! -f results/v2/checkpoints/v2_robots_r6i1_typed_s2.pt ]; then hold
  scripts/gpu_heavy.sh v2_robots_r6i1_typed_s2 "python -m physmem.v2_train train --data data/v2_r6_op1 --variant typed --run v2_robots_r6i1_typed_s2 --seed 2 --steps 8000 --bs 256 --eval-every 2000 --robot-balance --eval-splits test,heldout_robot" > logs/v2_robots_r6i1_typed_s2.log 2>&1; fi
# (2) competence gate of the round-1 models (VAL)
if [ ! -f results/v2/trust_r6i1.json ]; then hold
  nice -n 10 python -m physmem.v2_train eval --ckpt results/v2/checkpoints/v2_robots_r6i1_typed_s0.pt --data data/v2_r6_op1 --splits val > logs/v2_eval_r6i1_val.log 2>&1
  python scripts/v2/make_trust.py logs/v2_eval_r6i1_val.log results/v2/trust_r6i1.json v2_robots_r6i1_typed_s0; fi
# (3) round-2 on-policy rollouts (cells already written are skipped)
out=data/v2_r6_rollouts/round2; mkdir -p $out
cells=""
for rb in ur5e_2f85 panda; do for f in conveyor_sort bin_clearing singulation door_open tool_return pouch_transfer twist_lock machine_tending fixture_place; do cells="$cells $rb:$f"; done; done
cells="$cells probe:depalletize ur5e_suction:depalletize probe:conveyor_sort probe:singulation"
for c in $cells; do echo $c; done | xargs -P ${PROCS:-8} -I{} bash -c 'c={}; r=${c%%:*}; f=${c##*:}; [ -f '$out'/${r}__${f}.json ] && exit 0; while [ -f ~/dynagraph/HOLD ]; do sleep 60; done; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl nice -n 10 python scripts/v2/env_onpolicy.py 2 $r $f ${N_OP:-24} '$out' r6i1 results/v2/trust_r6i1.json >> logs/v2_r6_onpolicy_round2.log 2>&1'
echo ROUNDS_DONE
# (4) final stage
scripts/v2/r6_final.sh
# (5) paired videos: picked automatically from the final table (significantly positive cells, best first)
hold
nice -n 10 python scripts/v2/robot_table5.py r6f results/v2/closed_loop/robot_table5_r6f.md > logs/v2_table_r6f.log 2>&1
nice -n 10 python scripts/v2/pick_video_cells.py results/v2/closed_loop/robot_table5_r6f.json 4 > logs/v2_video_cells.txt
[ -s logs/v2_video_cells.txt ] && CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl nice -n 10 python scripts/v2/paired_videos.py r6f results/v2/trust_r6f.json $(cat logs/v2_video_cells.txt) --stills=1 > logs/v2_paired_videos.log 2>&1
echo RESUME_DONE
