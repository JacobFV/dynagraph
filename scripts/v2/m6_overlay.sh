#!/usr/bin/env bash
# MODEL-V2 round 6 runtime overlay (REMOTE side): ~/dynagraph_m6run71/src = frozen envs-r7.1 tree (~/dynagraph_r7_1/src)
# + ONLY MODEL-V2-owned files from ~/dynagraph/src. envs / skills / embodiments must stay byte-identical to the freeze.
set -e
R7=~/dynagraph_r7_1/src; M=~/dynagraph/src; O=~/dynagraph_m6run71/src
mkdir -p $O
rsync -a --delete --exclude __pycache__ $R7/ $O/
chmod -R u+w $O
rsync -a --exclude __pycache__ $M/physmem/models/v2/ $O/physmem/models/v2/
cp $M/physmem/planner.py $M/physmem/v2_train.py $O/physmem/
for d in envs skills embodiments; do diff -rq -x __pycache__ $R7/physmem/$d $O/physmem/$d; done
grep -q 'ENV_VERSION = "envs-r7.1"' $O/physmem/envs/__init__.py
echo "OVERLAY_OK envs-r7.1 $(date -u +%FT%TZ)"
