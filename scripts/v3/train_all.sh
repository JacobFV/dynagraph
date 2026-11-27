#!/usr/bin/env bash
# V3 §0 matched training runs, ONE gpu_small job at a time (the driver itself only waits). Run on gb10 from
# ~/dynagraph_v3:  nohup scripts/v3/train_all.sh > logs/train_all.log 2>&1 &
cd ~/dynagraph_v3
STEPS=${STEPS:-4000}; BS=${BS:-1024}
runs=(
  "lbs_law_s0 --kind lbs --seed 0"
  "v1_s0 --kind v1 --seed 0"
  "lbs_nolaw_s0 --kind lbs --nolaw --seed 0"
  "lbs_law_s1 --kind lbs --seed 1"
  "v1_s1 --kind v1 --seed 1"
  "lbs_nolaw_s1 --kind lbs --nolaw --seed 1"
  "lbs_law_s2 --kind lbs --seed 2"
  "v1_s2 --kind v1 --seed 2"
  "lbs_nolaw_s2 --kind lbs --nolaw --seed 2"
  "dense_law_s0 --kind dense --seed 0"
)
for spec in "${runs[@]}"; do
  set -- $spec; name=$1; shift
  [ -f results/v3/checkpoints/$name.pt ] && { echo "skip $name"; continue; }
  scripts/v3/gpu_job.sh $name -m physmem.v3_chain train --run $name --steps $STEPS --bs $BS --log-every 200 --eval-every 1000 "$@"
  while kill -0 $(cat logs/$name.pid) 2>/dev/null; do sleep 30; done
  grep -q "done" logs/$name.log && echo "[$(date -u +%T)] $name done" || { echo "[$(date -u +%T)] $name FAILED"; tail -5 logs/$name.log; }
done
echo TRAIN_ALL_DONE
