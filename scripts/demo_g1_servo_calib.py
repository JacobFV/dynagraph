"""Embodiment-adapter calibration of the push stroke on TRAINING calibration seeds (10000+, never the
test seeds 0-49). Measures (eval-only, sim truth) the hand release-speed ratio and the along-push error
of the ORACLE prediction (true friction) over N single strokes. Usage:
  demo_g1_servo_calib.py --servo v1|v2 [--gains json] [--n 50] [--out results/g1/servo_calib_<name>.json]"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from physmem.robots import g1
from physmem import control as C
from physmem.schema import ActionCandidate, EventTensors, QueryTensors, Prop, HORIZONS

ap = argparse.ArgumentParser()
ap.add_argument("--servo", default="v2"); ap.add_argument("--gains", default="{}"); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", default=None); ap.add_argument("--seed0", type=int, default=10000)
a = ap.parse_args()
g1.PUSH_SERVO = a.servo
g1.SERVO_GAINS.update(json.loads(a.gains))

def oracle_along(mu, half, params):
    stub = C.PriorCoulombStub(mu)
    ev = EventTensors(entity=torch.zeros(1, 1, dtype=torch.long), related=torch.full((1, 1), -1), prop=torch.full((1, 1), int(Prop.SIZE)),
                      value=torch.tensor([[list(half) + [0.0] * 5]], dtype=torch.float32), logvar=torch.zeros(1, 1, 8),
                      age=torch.zeros(1, 1), latency=torch.zeros(1, 1), source=torch.zeros(1, 1, dtype=torch.long),
                      provenance=torch.zeros(1, 1, dtype=torch.long), mask=torch.ones(1, 1, dtype=torch.bool),
                      ent_type=torch.zeros(1, 4, dtype=torch.long), ent_mask=torch.ones(1, 4, dtype=torch.bool), dt_obs=torch.tensor([0.05]))
    c = ActionCandidate("o", "push", tuple(params), g1.OBJ)
    q = QueryTensors(target=torch.zeros(1, dtype=torch.long), action=torch.tensor(c.vector()[None]), horizons=torch.tensor([HORIZONS], dtype=torch.float32))
    return stub(ev, q).mean[0, -1, :2].numpy()

rows = []
rng = np.random.default_rng(a.seed0)
for k in range(a.n):
    seed = a.seed0 + k
    ep = g1.sample_slide_episode(seed)
    w = g1.G1PushWorld(ep, seed=seed); r = w.actor; sim = w._sim
    sp = float(np.exp(rng.uniform(np.log(0.1), np.log(0.8))))
    du = float(np.clip(min(rng.uniform(0.1, 0.5), 0.08 / sp), 0.1, 0.5)); sp = min(sp, 0.08 / du)
    st0 = w.true_object(); yaw = st0["yaw"]
    gx, gy = ep.target; ang = math.atan2(gy - st0["pos"][1], gx - st0["pos"][0]) - yaw
    params = (ang + math.pi + float(rng.normal(0, 0.2)), float(rng.normal(0, 0.1)), sp, du)
    cand = ActionCandidate("cal", "push", params, g1.OBJ)
    r.hold(0.3); r.drain_events()
    pose = (st0["pos"][0], st0["pos"][1], math.cos(yaw), math.sin(yaw))
    plan = r.plan_push(cand, pose, ep.box.half)
    if not plan["feasible"]:
        continue
    hp = []
    orig = r._tick
    def tick(t, phase=""):
        orig(t, phase)
        sid = sim.model.site(f"{r.active_side}_pusher").id
        hp.append((phase, sim.data.site_xpos[sid].copy()))
    r._tick = tick
    info = r.execute_push(cand, pose, ep.box.half, plan)
    r._tick = orig
    r.hold(2.0)
    d = np.array(info["dir"])
    P = [(ph, x) for ph, x in hp]
    idx = [i for i, (ph, _) in enumerate(P) if ph == "push"]
    if len(idx) < 3:
        continue
    xs = np.array([P[i][1][:2] for i in idx])
    v = np.diff(xs, axis=0) @ d / g1.CONTROL_DT
    v_rel = float(v[-3:].mean())
    st1 = w.true_object()
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    real = R.T @ (st1["pos"][:2] - st0["pos"][:2])
    pred = oracle_along(ep.box.friction, ep.box.half, params)
    u = pred / max(np.linalg.norm(pred), 1e-9)
    rows.append(dict(seed=seed, speed=sp, dur=du, mu=ep.box.friction, mass=ep.box.mass, release_ratio=v_rel / sp,
                     stroke_real=float(np.linalg.norm(xs[-1] - xs[0])), pred_along=float(pred @ u), real_along=float(real @ u),
                     err_along=float(real @ u - pred @ u), err_lat=float(abs(real @ np.array([-u[1], u[0]])))))
rr = np.array([x["release_ratio"] for x in rows]); ea = np.array([x["err_along"] for x in rows])
fast = np.array([x["speed"] >= 0.45 for x in rows])
summ = dict(servo=a.servo, gains=g1.SERVO_GAINS if a.servo == "v2" else None, n=len(rows),
            release_ratio_p10=float(np.percentile(rr, 10)), release_ratio_p50=float(np.median(rr)), release_ratio_p90=float(np.percentile(rr, 90)),
            err_along_mean=float(ea.mean()), err_along_std=float(ea.std()), err_along_std_fast=float(ea[fast].std()) if fast.any() else None,
            n_fast=int(fast.sum()), err_lat_mean=float(np.mean([x["err_lat"] for x in rows])),
            corr_err_mu=float(np.corrcoef(ea, [x["mu"] for x in rows])[0, 1]),
            corr_err_mass=float(np.corrcoef(ea, [x["mass"] for x in rows])[0, 1]),
            calibration_seeds=f"{a.seed0}..{a.seed0 + a.n - 1} (training-calibration only)")
print(json.dumps(summ))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, strokes=rows), indent=1))
