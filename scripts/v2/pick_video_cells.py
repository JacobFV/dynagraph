"""pick up to K robot:family:seed cells for paired videos: cells where residual_gated (or residual_gated_probe) is
significantly better than the plain heuristic (CI lower bound > 0), best Δ first; seed = the first paired test task
whose outcome differs in the policy's favour (pre-registered rule: lowest seed index with heuristic fail / planner success).
output cells robot:family:seed:split:idx (idx = paired-task index -> same planner rng as in the table).
usage: pick_video_cells.py <robot_table5.json> <K>"""
import glob, json, sys
summ = json.load(open(sys.argv[1])); K = int(sys.argv[2])
cands = []
for key, v in summ.items():
    robot, fam, g = key.split("|")
    if g in ("residual_gated", "residual_gated_probe") and "ci_heur" in v and v["ci_heur"][0] > 0:
        cands.append((v["d_heur"], robot, fam, g))
cands.sort(reverse=True)
seen, out = set(), []
for d, robot, fam, g in cands:
    if (robot, fam) in seen:
        continue
    seen.add((robot, fam))
    f = glob.glob(f"results/v2/closed_loop/v2_tab5_r6f_{robot}_{fam}.json")
    if not f:
        continue
    ep = json.load(open(f[0]))[fam]["episodes"]
    pol = "planner_" + g
    idx = next((i for i, (h, p) in enumerate(zip(ep["heuristic"], ep[pol])) if not h["success"] and p["success"]), 0)
    split = "heldout_robot" if robot == "gen3_2f85" else "test"
    seed = {"test": 2_000_000, "heldout_robot": 2_000_000}.get(split, 2_000_000) + 900_000 + idx
    out.append(f"{robot}:{fam}:{seed}:{split}:{idx}:{g}")
    if len(out) >= K:
        break
print(" ".join(out))
