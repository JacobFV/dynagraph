#!/usr/bin/env bash
# round-6 DRY RUN (not pre-registered), envs-r6 (asserted): il-v0 demos, scripted policy, train split, 60 per cell
# (+ the memory-model view of the same episodes), then scripted VAL baselines (20 per cell, seeds 2950000+i). 2 procs.
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl EXPECT_ENV=envs-r6
rec() { [ -f data/v2_il_dry/$1__$2.npz ] || nice -n 10 python scripts/v2/il_record.py $1 $2 60 data/v2_il_dry >> logs/v2_il_dry_record.log 2>&1
        [ -f data/v2_il_dry_val/$1__$2.npz ] || SPLIT=val nice -n 10 python scripts/v2/il_record.py $1 $2 20 data/v2_il_dry_val 2950000 >> logs/v2_il_dry_record.log 2>&1; }
export -f rec
printf '%s\n' "fixture_place ur5e_2f85" "fixture_place panda" "twist_lock ur5e_2f85" "twist_lock panda" "door_open ur5e_2f85" "door_open panda" | xargs -P 2 -L 1 bash -c 'rec $0 $1'
echo IL_DRY_RECORD_DONE >> logs/v2_il_dry_record.log
