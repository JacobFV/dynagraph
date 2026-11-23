"""Achieved hand speed vs commanded push speed for the G1 push primitive (with the box
being pushed), plus friction sensitivity at each speed. Writes results/g1/speed_tracking.json."""
import json, math, sys
from pathlib import Path
import numpy as np
from physmem.robots import g1
from physmem.schema import ActionCandidate

rows = []
speeds = [float(s) for s in sys.argv[1:]] or [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
for sp in speeds:
    du = float(np.clip(0.10 / sp, 0.08, 0.6))  # ~10 cm commanded travel
    for th in (math.pi, math.pi / 2):
        for mu in (0.3, 0.7):
            ep = g1.sample_push_episode(11, friction=mu, mass=0.6)
            ep.box.pos, ep.box.yaw = (0.36, -0.04), 0.2
            w = g1.G1PushWorld(ep, seed=0); r = w.actor; sim = w._sim
            sid = sim.model.site(f"right_pusher").id
            vel = []
            orig = r._tick
            def tick(t, phase=""):
                orig(t, phase)
                if phase == "push":
                    sidx = sim.model.site(f"{r.active_side}_pusher").id
                    vel.append(sim.data.site_xpos[sidx].copy())
            r._tick = tick
            st0 = w.true_object()
            pose = (st0["pos"][0], st0["pos"][1], math.cos(st0["yaw"]), math.sin(st0["yaw"]))
            cand = ActionCandidate("s", "push", (th, 0.0, sp, du), g1.OBJ)
            plan = r.plan_push(cand, pose, ep.box.half)
            info = r.execute_push(cand, pose, ep.box.half, plan)
            r.hold(2.0)
            st1 = w.true_object()
            P = np.array(vel)
            v = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) / g1.CONTROL_DT
            n = len(v)
            mid = v[n // 3: max(n // 3 + 1, 2 * n // 3 + 1)]
            travel = float(np.linalg.norm(P[-1, :2] - P[0, :2]))
            tr = w.trace()
            at_stop = min(tr, key=lambda z: abs(z["t"] - info["t_push_end"]))
            rows.append(dict(cmd_speed=sp, duration=du, theta=th, friction=mu, feasible=plan["feasible"],
                             achieved_mid_speed=float(mid.mean()), achieved_peak=float(v.max()),
                             travel_cmd=sp * du, travel_achieved_in_window=travel,
                             disp=float(np.linalg.norm(st1["pos"][:2] - st0["pos"][:2])),
                             post_stop_slide=math.hypot(st1["pos"][0] - at_stop["x"], st1["pos"][1] - at_stop["y"]),
                             final_xy=st1["pos"][:2].tolist(), max_box_speed=w.audit()["max_obj_speed"],
                             tilt_z=float(st1["pos"][2] - st0["pos"][2]), clean=w.push_success()["clean"]))
            x = rows[-1]
            print(f"v {sp:4.2f} T {du:.2f} th {th:4.2f} mu {mu}: hand mid {x['achieved_mid_speed']:.3f} peak {x['achieved_peak']:.3f} "
                  f"ratio {x['achieved_mid_speed']/sp:.2f} | box disp {x['disp']*100:5.1f} slide {x['post_stop_slide']*100:5.1f} cm dz {x['tilt_z']*1000:+.1f}mm", flush=True)
print("\nfriction effect (|final xy(mu .3) - final xy(mu .7)|):")
for sp in speeds:
    for th in (math.pi, math.pi / 2):
        g = [x for x in rows if x["cmd_speed"] == sp and x["theta"] == th]
        print(f"  v {sp:4.2f} th {th:4.2f}: {math.dist(g[0]['final_xy'], g[1]['final_xy'])*100:5.2f} cm   tracking {np.mean([x['achieved_mid_speed'] for x in g])/sp:.2f}")
Path("results/g1").mkdir(parents=True, exist_ok=True)
Path("results/g1/speed_tracking.json").write_text(json.dumps(rows, indent=1))
