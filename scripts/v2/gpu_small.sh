#!/usr/bin/env bash
# MODEL-V2 small GPU jobs, concurrent with INKLING's heavy lock (lead decision, round 6): own lock
# ~/dynagraph/.gpu_small.lock (one small job at a time), niced, waits until >= 40 GB RAM is available.
# Jobs must stay < 6 GB. usage: gpu_small.sh <name> <cmd...>
name=$1; shift
mkdir -p ~/dynagraph/logs
exec flock ~/dynagraph/.gpu_small.lock bash -c "until [ \$(free -g | awk '/Mem:/ {print \$7}') -ge 40 ]; do sleep 60; done; echo \"[\$(date -u +%FT%TZ)] START $name\" >> ~/dynagraph/logs/gpu_small_ledger.log; $*; rc=\$?; echo \"[\$(date -u +%FT%TZ)] END $name rc=\$rc\" >> ~/dynagraph/logs/gpu_small_ledger.log; exit \$rc"
