#!/usr/bin/env bash
# full family x robot matrix (5 seeds) on the current envs; appends to $1 (default results/v2/robots/env_runs_r5.jsonl)
cd ~/dynagraph && source .venv/bin/activate && export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1
F=${1:-results/v2/robots/env_runs_r5.jsonl}
M=scripts/robots_env_matrix.py
python $M --robots panda,ur5e_2f85,gen3_2f85,aloha,tidybot --fams all --seeds 0-4 --procs 5 --guard 300 --out $F
python $M --robots gantry_vacuum --fams depalletize,conveyor_sort --seeds 0-4 --procs 5 --guard 400 --out $F
python $M --robots gantry_clamp --fams tote_transfer,conveyor_sort --seeds 0-4 --procs 5 --guard 400 --out $F
python $M --robots ur5e_suction,ur5e_suction2,fr3_suction --fams depalletize,singulation,fixture_place --seeds 0-4 --procs 5 --guard 300 --out $F
python $M --robots z1,so100 --fams small_parts_pick,small_peg_insert,singulation,drawer_open,door_open --seeds 0-4 --procs 5 --guard 300 --out $F
python $M --robots g1_dex3 --fams singulation,door_open,drawer_open,fixture_place --seeds 0-4 --procs 2 --guard 600 --out $F
