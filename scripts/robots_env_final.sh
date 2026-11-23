#!/usr/bin/env bash
# final family x robot matrix (5 seeds); appends to results/v2/robots/env_runs_final.jsonl
cd ~/dynagraph && source .venv/bin/activate && export PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1
F=results/v2/robots/env_runs_final.jsonl
M=scripts/robots_env_matrix.py
python $M --robots panda,ur5e_2f85,gen3_2f85,aloha,tidybot --fams all --seeds 0-4 --procs 5 --guard 300 --out $F
python $M --robots ur5e_suction,fr3_suction --fams depalletize,singulation,bulk_sweep,pouch_transfer,fixture_place --seeds 0-4 --procs 5 --guard 300 --out $F
python $M --robots z1,so100 --fams singulation,fixture_place,peg_insert,drawer_open,door_open,pouch_transfer,bin_clearing --seeds 0-4 --procs 5 --guard 300 --out $F
