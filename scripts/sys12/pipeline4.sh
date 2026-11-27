#!/usr/bin/env bash
# after pipeline3: shape-invariant prefix gate test + free-form language eval with a 256-token budget. Frozen tree.
cd ~/dynagraph_sys12 && source ~/dynagraph/.venv/bin/activate
export PYTHONPATH=$HOME/dynagraph_sys12/src PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "[p]ipeline3.sh" > /dev/null; do sleep 120; done
hold() { while [ -f ~/dynagraph/HOLD ] || [ -f ~/dynagraph/HOLD_SYS12 ]; do sleep 60; done; }
CK=results/sys12/checkpoints
hold; [ -f results/sys12/gate_test_rowwise.json ] || scripts/gpu_heavy.sh sys12_gate_rowwise "nice -n 10 python scripts/sys12/gate_test_rowwise.py --ckpt $CK/s1r1_inkling_gated_s0.pt" > logs/sys12/gate_rowwise.log 2>&1
hold; [ -f results/sys12/lang_eval_gen256.json ] || scripts/gpu_heavy.sh sys12_lang256 "nice -n 10 python scripts/sys12/lang_eval.py --gated $CK/s1r1_inkling_gated_s0.pt --ungated $CK/s1r1_inkling_ungated_s0.pt --no-ppl --gen-tokens 256 --gated-subset 12 --out results/sys12/lang_eval_gen256.json" > logs/sys12/lang_eval256.log 2>&1
nice -n 10 python scripts/sys12/report.py > logs/sys12/report.log 2>&1
echo SYS12_PIPELINE4_DONE
