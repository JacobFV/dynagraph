"""Hidden-friction sensitivity of the G1 push primitive: same initial state, same
candidate, friction 0.3..0.7 -> final box displacement and post-stop slide.
Writes results/g1/friction_sweep.json."""
import json, math
from pathlib import Path
import numpy as np
from physmem.robots import g1
from physmem.schema import ActionCandidate

rows = []
for th, sp, du in ((math.pi, 0.4, 0.3), (math.pi / 2, 0.4, 0.3), (2.6, 0.3, 0.4), (0.0, 0.4, 0.25)):
    for mu in (0.3, 0.4, 0.5, 0.6, 0.7):
        ep = g1.sample_push_episode(11, friction=mu, mass=0.6)
        ep.box.pos, ep.box.yaw = (0.40, -0.04), 0.2
        w = g1.G1PushWorld(ep, seed=0)
        r = w.actor
        st0 = w.true_object()
        pose = (st0["pos"][0], st0["pos"][1], math.cos(st0["yaw"]), math.sin(st0["yaw"]))
        cand = ActionCandidate("sweep", "push", (th, 0.0, sp, du), g1.OBJ)
        info = r.execute_push(cand, pose, ep.box.half)
        r.hold(1.5)
        st1 = w.true_object()
        tr = w.trace()
        at_stop = min(tr, key=lambda z: abs(z["t"] - info["t_push_end"]))
        slide = math.hypot(st1["pos"][0] - at_stop["x"], st1["pos"][1] - at_stop["y"])
        rows.append(dict(theta=th, speed=sp, duration=du, friction=mu, final_xy=st1["pos"][:2].tolist(),
                         disp=float(np.linalg.norm(st1["pos"][:2] - st0["pos"][:2])), post_stop_slide=slide,
                         dyaw=st1["yaw"] - st0["yaw"], audit_clean=w.push_success()["clean"]))
        print(f"th {th:4.2f} v {sp} T {du} mu {mu}: disp {rows[-1]['disp']*100:5.2f} cm  slide-after-stop {slide*100:5.2f} cm  dyaw {rows[-1]['dyaw']:+.3f}")
for th in sorted(set(r["theta"] for r in rows)):
    g = [r for r in rows if r["theta"] == th]
    a, b = g[0], g[-1]
    print(f"theta {th:4.2f}: final-position difference mu0.3 vs mu0.7 = {math.dist(a['final_xy'], b['final_xy'])*100:.2f} cm")
Path("results/g1").mkdir(parents=True, exist_ok=True)
Path("results/g1/friction_sweep.json").write_text(json.dumps(rows, indent=1))
