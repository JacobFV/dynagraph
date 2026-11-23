"""Evaluate all trained inkling-main runs (best-val adapters) with the shared metric code."""
import json, sys
import yaml
sys.path.insert(0, "src")
from physmem.models.inkling import evaluate_inkling
cfg = yaml.safe_load(open("configs/sprint.yaml"))
rows = evaluate_inkling(cfg)
print("rows", len(rows))
