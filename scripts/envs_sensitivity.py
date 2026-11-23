"""Hidden-variable sensitivity: for a FIXED action, how much does the outcome change across the hidden
variable's range? (v1's mistake was a hidden variable that barely mattered.)

    python scripts/envs_sensitivity.py [families|all] [n_seeds] [procs]  -> results/v2/envs/sensitivity.json

Each protocol sets the hidden variable through TaskSpec.params (evaluator-only overrides), runs the same
fixed skill sequence (no retries, no adaptation) and records a scalar outcome. Robot = test-only probe.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from physmem.envs import make_env
from physmem.v2.api import SkillCall, TaskSpec


def _reset(fam, seed, params):
    env = make_env(fam)
    env.reset(TaskSpec(fam, env.category, seed, "probe", "sens", params))
    return env


def _first(env, kind=None):
    return sorted(env.objects(kind))[0]


def p_bin_clearing(seed, v):
    var, val = v
    hid = {"mass": 0.5, "mu_grip": 0.5}
    hid[var] = val
    env = _reset("bin_clearing", seed, dict(n_parts=1, hidden=hid, shapes=("box", "cylinder")))
    out = env.execute(SkillCall("pick", dict(grip_force=15.0, lift_speed=0.3, depth=0.5), target=_first(env)))
    return float(out.result.ok)


def p_singulation(seed, mu):
    env = _reset("singulation", seed, dict(n_objects=1, hidden={"mu": mu, "mass": 1.0}))
    e = _first(env)
    p0 = env.true_pose(e)[0]
    env.execute(SkillCall("push", dict(theta=math.pi, phi=0.0, speed=0.6, distance=0.05, height=0.012), target=e))
    env.run(1.0)
    return float(np.linalg.norm(env.true_pose(e)[0][:2] - p0[:2]))


def p_bulk_sweep(seed, mu):
    env = _reset("bulk_sweep", seed, dict(mu_mean=mu, n_parts=60, target_x=0.15))
    c = env.scripted()
    c = SkillCall("sweep", dict(c.params, speed=0.5, stop_time=0.1), target=c.target)
    env.execute(c)
    return float(env.progress())


def p_depalletize(seed, m):
    env = _reset("depalletize", seed, dict(layers=1, hidden={"mass": m}, heavy_p=0.0))
    out = env.execute(SkillCall("pick", dict(suction=0.5, lift_speed=0.25, lift_height=0.1), target="carton0"))
    return float(out.result.ok)


def p_conveyor_sort(seed, v):
    env = _reset("conveyor_sort", seed, dict(belt_speed=v, n_parts=1, cycle_s=0.0))
    env.idle(max(0.0, (-0.3 - (-0.75)) / v))     # wait until the part is in the window
    e = _first(env)
    out = env.execute(SkillCall("pick", dict(grip_force=25.0, lift_speed=0.3, depth=0.4, vx=0.075, vy=0.0,
                                             approach_speed=0.6, lift_height=0.12), target=e))
    return float(out.result.ok)


def p_tote_transfer(seed, m):
    env = _reset("tote_transfer", seed, dict(n_totes=1, content_mass=m, hidden={"mu_grip": 0.6}))
    out = env.execute(SkillCall("pick", dict(grip_force=40.0, lift_speed=0.25, depth=0.5, lift_height=0.15),
                                target="tote0"))
    return float(out.result.ok)


def _insert_episode(fam, seed, params, ins):
    env = _reset(fam, seed, params)
    env.execute(env._pick(env.obj, grip_force=30.0) if fam != "tool_return" else env._pick(env.obj, grip_force=ins.pop("grip", 30.0)))
    if env.held is None:
        return 0.0
    env.execute(env._insert(env.region_name, **ins))
    return float(env.success())


def p_peg_insert(seed, v):
    clr, ch = v
    return _insert_episode("peg_insert", seed, dict(clearance=clr, chamfer=ch, mu=0.5),
                           dict(search_r=0.0, down_force=6.0, compliance=0.5, timeout=3.0))


def p_connector_insert(seed, v):
    clr, mu = v
    return _insert_episode("connector_insert", seed, dict(clearance=clr, chamfer=0.0, mu=mu),
                           dict(search_r=0.0, wiggle=0.0, down_force=15.0, compliance=0.5, timeout=3.0))


def p_tool_return(seed, m):
    """in-hand tilt (deg) of the tool after pick + 1 s carry: the hidden CoM torque rotates it in the grasp."""
    env = _reset("tool_return", seed, dict(head_mass=m, head_imbalance=0.5, mu_grip=0.4, clearance=0.0015))
    env.execute(env._pick(env.obj, grip_force=12.0, lift_speed=0.4))
    if env.held is None:
        return float("nan")
    reg = env.region("holder")
    env.execute(SkillCall("reach", dict(x=reg["center"][0], y=reg["center"][1], z=env.safe_z, speed=0.6, dwell=0.5)))
    q = env.true_pose("tool")[1]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))


def p_fixture_place(seed, v):
    var, val = v
    hid = {"mu_grip": 0.6, "mass": 1.0, "mu": 0.4}
    hid[var] = val
    env = _reset("fixture_place", seed, dict(mass=hid["mass"], mu=hid["mu"], mu_grip=hid["mu_grip"]))
    env.execute(env._pick("fixture", depth=0.0, grip_force=20.0, lift_speed=0.5))
    if env.held is None:
        return 0.0
    env.execute(env._place(speed=0.8, release_h=0.01, lower_speed=0.2))
    return float(env.success())


def p_kitting_pocket(seed, clr):
    env = _reset("kitting_pocket", seed, dict(n_parts=1, clearance=clr))
    env.execute(env._pick("part0", grip_force=30.0))
    if env.held is None:
        return 0.0
    env.execute(SkillCall("place", dict(release_h=0.004, speed=0.3, lower_speed=0.05, align=1, correct=1),
                          target="pocket_part0"))
    return float(env.seated("part0"))


def p_drawer_open(seed, fl):
    env = _reset("drawer_open", seed, dict(frictionloss=fl, damping=20.0, mass=5.0, mu_grip=0.6))
    env.execute(env._open(grip_force=15.0, travel=0.2, speed=0.15))
    return float(env.q())


def p_door_open(seed, fl):
    env = _reset("door_open", seed, dict(frictionloss=fl, damping=0.5, mass=3.0, mu_grip=0.6))
    env.execute(env._open(grip_force=8.0, travel=1.4, speed=0.15))
    return float(env.q())


def p_pouch_transfer(seed, y):
    env = _reset("pouch_transfer", seed, dict(young=y, mass=0.3))
    env.execute(SkillCall("pick", dict(grip_force=12.0, lift_speed=0.2, depth=0.0), target="pouch"))
    if env.held is None:
        return 0.0
    env.execute(SkillCall("place", dict(release_h=0.03, speed=0.3), target="tote"))
    return float(env.success())


PROTOCOLS = {
    "bin_clearing": [("mass_kg|mu_grip=0.5, grip 15 N", p_bin_clearing, [("mass", m) for m in (0.1, 0.5, 1.0, 1.5, 2.5)], "P(hold)"),
                     ("mu_grip|mass=0.5 kg, grip 15 N", p_bin_clearing, [("mu_grip", m) for m in (0.08, 0.15, 0.3, 0.6, 1.0)], "P(hold)")],
    "singulation": [("mu (object<->bench)|push 0.6 m/s x 5 cm", p_singulation, [0.15, 0.3, 0.5, 0.7, 0.9], "displacement_m")],
    "bulk_sweep": [("mu_mean|sweep 0.5 m/s, stop 0.1 s", p_bulk_sweep, [0.12, 0.25, 0.4, 0.6, 0.8], "fraction_in_target")],
    "depalletize": [("carton mass kg|suction 0.5 (150 N), lift 0.25 m/s", p_depalletize, [2.0, 6.0, 10.0, 13.0, 16.0, 22.0], "P(hold)")],
    "conveyor_sort": [("belt speed m/s|intercept assumes 0.075", p_conveyor_sort, [0.03, 0.05, 0.075, 0.1, 0.12], "P(pick)")],
    "tote_transfer": [("content mass kg|grip 40 N, mu_g 0.6", p_tote_transfer, [0.5, 2.0, 4.0, 7.0, 10.0, 14.0], "P(hold)")],
    "peg_insert": [("(clearance m, chamfer m)|straight insert, no search", p_peg_insert,
                    [(c, ch) for ch in (0.0, 0.002) for c in (0.0005, 0.001, 0.0015, 0.002)], "P(success)")],
    "connector_insert": [("(clearance m, mu)|straight, no wiggle, 15 N", p_connector_insert,
                          [(c, mu) for mu in (0.3, 1.0) for c in (0.0005, 0.001, 0.0015)], "P(success)")],
    "tool_return": [("head mass kg|imbalance 0.5, grip 12 N, mu_g 0.4, carry", p_tool_return, [0.1, 0.3, 0.5, 0.7, 0.9], "in-hand tilt deg")],
    "fixture_place": [("mu_grip|mass 1 kg, grip 20 N, fast carry 0.8 m/s", p_fixture_place, [("mu_grip", m) for m in (0.15, 0.25, 0.4, 0.7, 1.0)], "P(+-2mm)"),
                      ("mass kg|mu_g 0.6, grip 20 N, fast carry", p_fixture_place, [("mass", m) for m in (0.3, 1.0, 2.0, 3.0, 4.0)], "P(+-2mm)")],
    "kitting_pocket": [("pocket clearance m|plain place (no compliant insert)", p_kitting_pocket, [0.001, 0.0015, 0.002, 0.003, 0.004], "P(seated)")],
    "drawer_open": [("frictionloss N|grip 15 N, mu_g 0.6", p_drawer_open, [2.0, 5.0, 10.0, 20.0, 40.0], "drawer travel m")],
    "door_open": [("hinge frictionloss N m|grip 8 N, mu_g 0.6", p_door_open, [0.2, 0.5, 1.0, 2.0, 4.0], "door angle rad")],
    "pouch_transfer": [("Young modulus Pa|grip 12 N", p_pouch_transfer, [2e3, 6e3, 2e4, 6e4], "P(success)")],
}


def _job(args):
    fam, pi, val, seed = args
    name, fn, _, _ = PROTOCOLS[fam][pi]
    t0 = time.time()
    try:
        out = fn(seed, val)
        err = None
    except Exception as e:      # keep every run, including failures
        out, err = float("nan"), repr(e)
    return dict(family=fam, protocol=name, value=val, seed=seed, outcome=out, error=err, wall=time.time() - t0)


def main():
    fams = list(PROTOCOLS) if len(sys.argv) < 2 or sys.argv[1] == "all" else sys.argv[1].split(",")
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    jobs = [(f, i, v, 1000 + s) for f in fams for i, p in enumerate(PROTOCOLS[f]) for v in p[2] for s in range(n_seeds)]
    t0 = time.time()
    with Pool(procs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    os.makedirs("results/v2/envs", exist_ok=True)
    summary = []
    for f in fams:
        for i, (name, _, vals, label) in enumerate(PROTOCOLS[f]):
            pts = []
            for v in vals:
                o = np.array([r["outcome"] for r in rows if r["family"] == f and r["protocol"] == name and r["value"] == v])
                pts.append(dict(value=v, mean=float(np.nanmean(o)), std=float(np.nanstd(o)), n=int(np.isfinite(o).sum())))
            means = [p["mean"] for p in pts]
            summary.append(dict(family=f, protocol=name, outcome=label, points=pts,
                                range=float(np.nanmax(means) - np.nanmin(means))))
            print(f"{f:16s} {name:55s} {label:18s} " + " ".join(f"{p['value']}:{p['mean']:.3f}" for p in pts))
    errs = [r for r in rows if r["error"]]
    for r in errs[:10]:
        print("ERROR", r)
    out = dict(rows=rows, summary=summary, n_seeds=n_seeds, wall_s=time.time() - t0, robot="probe (test-only)")
    path = "results/v2/envs/sensitivity.json" if len(fams) == len(PROTOCOLS) else f"results/v2/envs/sensitivity_{'_'.join(fams)[:60]}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path, "wall", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
