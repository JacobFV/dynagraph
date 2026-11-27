#!/usr/bin/env bash
# r9 FULL-data regime: r2-r6 (+tranche8, rollouts; data/v2_r7_final) + tranche9_r7 + tranche9_r7_1_gen3 + the il-v1
# demo episodes (memory view) -> data/v2_r9_full; then the compute-matched grid (r9_mem_train.sh REGIME=full).
cd ~/dynagraph && source .venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_m6run71/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
while pgrep -f "[v]2_train convert" > /dev/null; do sleep 60; done
[ -f data/v2_r9_t9/manifest.json ] || { echo "t9 convert missing"; exit 1; }
[ -d data/v2_r9_fullbase ] || nice -n 10 python scripts/v2/merge_data.py data/v2_r9_fullbase data/v2_r7_final data/v2_r9_t9 data/v2_r9_t9g3 > logs/v2_r9_merge_full.log 2>&1
[ -d data/v2_r9_full ] || nice -n 10 python scripts/v2/merge_onpolicy.py data/v2_r9_fullbase data/v2_r9_full data/v2_il_r7/mem >> logs/v2_r9_merge_full.log 2>&1
du -sh data/v2_r9_full >> logs/v2_r9_merge_full.log
REGIME=full DATA=data/v2_r9_full scripts/v2/r9_mem_train.sh
