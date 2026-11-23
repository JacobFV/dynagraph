"""Gate: 10 seeded rail lift/place controller trials (G1 bimanual squeeze).
Writes results/g1/rail_gate.json and optional snapshots."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from physmem.robots import g1
from physmem.schema import Prop

seeds = [int(s) for s in sys.argv[1:]] or list(range(10))
rows = []
for seed in seeds:
    ep = g1.sample_rail_episode(seed)
    w = g1.G1RailWorld(ep, seed=seed)
    r = w.actor
    r.hold(0.3, "observe")
    evs = r.drain_events()
    pose = [e for e in evs if e.entity_id == g1.OBJ and e.property_type == Prop.POSE2][-1].value
    size = [e for e in evs if e.entity_id == g1.OBJ and e.property_type == Prop.SIZE][-1].value
    t0 = time.perf_counter()
    err = None
    try:
        log = r.lift_place(pose, size, ep.tray_xy, z_support=g1.TABLE_TOP + ep.rail_height)
        r.hold(0.8, "final")
    except g1.PhysicsIntegrityError as e:
        log, err = {}, str(e)
    res = w.lift_place_success()
    res.update(seed=seed, mass=ep.box.mass, friction=ep.box.friction, half=ep.box.half, wall=time.perf_counter() - t0,
               ik=log, error=err, audit=w.audit())
    rows.append(res)
    print(f"seed {seed}: success {res['success']} clear {res['max_clearance']*100:.1f}cm dist {res['dist']*100:.1f}cm still {res['still']} "
          f"supported {res['supported']} released {res['released']} clean {res['clean']} stable {res['stable']} t {res['sim_time']:.1f}s "
          f"m {ep.box.mass:.2f} ik {max(log.values()) if log else -1:.3f}", flush=True)
n = sum(r["success"] for r in rows)
print(f"GATE: {n}/{len(rows)} lift/place successes")
Path("results/g1").mkdir(parents=True, exist_ok=True)
Path("results/g1/rail_gate.json").write_text(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                            n_success=n, n=len(rows), trials=rows), indent=1, default=float))
