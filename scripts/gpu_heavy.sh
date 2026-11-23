#!/usr/bin/env bash
# REMOTE-side wrapper: serialize heavy GPU jobs. usage (on gb10-direct): scripts/gpu_heavy.sh <name> <cmd...>
name=$1; shift
mkdir -p ~/dynagraph/logs
exec flock ~/dynagraph/.gpu_heavy.lock bash -c "echo \"[\$(date -u +%FT%TZ)] START $name\" >> ~/dynagraph/logs/gpu_ledger.log; $*; rc=\$?; echo \"[\$(date -u +%FT%TZ)] END $name rc=\$rc\" >> ~/dynagraph/logs/gpu_ledger.log; exit \$rc"
