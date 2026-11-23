"""Generate the G1 calibration/adapter dataset "g1cal" (experiment e15) in the PhysSplit tensor format of
physmem.data (load_split("train"|"val"|"test", "g1cal")).

- Roots = slide-to-target episodes on CALIBRATION seeds (default 20000+; disjoint from every G1 test seed
  0-49 and the servo-calibration seeds 10000-10049). Split BY ROOT before windowing:
  train = seeds [s0, s0+n_train), val = next n_val, test = next n_test.
- Per root: up to K=4 random-but-feasible push primitives drawn from the slide-to-target candidate
  distribution (control.generate_candidates with fast_frac=0.4, 8 cm stroke cap, duration 0.1-0.5 s),
  IK-checked on the robot-only model, executed by controller v1. Hidden friction U[0.3,0.7], mass
  log-U[0.2,2] kg, as in slide-to-target.
- Sample k = (actor event stream causally available at the decision time t_dec (before the approach),
  action k, future box state at t_push + HORIZONS in the canonical frame of the actor's latest pose
  estimate at t_dec) -- exactly the query the G1 controller makes at run time.
- After each stroke the robot holds until t_push + 2.1 s, so every horizon is observed before the next
  approach starts (future_mask all True unless the box left the table).
- Labels follow data.make_labels_np: contact_mode from hand force / tilt / speed, free_slide_mask after
  stroke end + 0.05 s with no hand/other contact, flat, moving at free-phase start. law = 0 (Coulomb).
- Scales are NOT refit: scales.json is a byte copy of data/v2/scales.json (fine-tuning keeps v2 scalers).
Ground truth is recorded at 100 Hz by an evaluator-side recorder (never actor input).

  python scripts/demo_g1_make_g1cal.py --n-train 600 --n-val 100 --n-test 100 --procs 8
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

K_MAX = 4
ROOT_BASE = 9_000_000  # root ids = ROOT_BASE + seed (disjoint from data.py root ints)
HOLD_AFTER_PUSH = 2.1


def gen_root(seed):
    import torch  # noqa: F401  (tensorize helpers import lazily)
    from physmem import control as C
    from physmem import data as D
    from physmem.robots import g1
    from physmem.schema import HORIZONS, Prop

    t0 = time.time()
    ep = g1.sample_slide_episode(seed)
    w = g1.G1PushWorld(ep, seed=seed)
    sim, r = w._sim, w.actor
    rng = np.random.default_rng(seed + 777)
    rec = []  # 100 Hz truth (evaluator side)

    orig = r._tick

    def tick(t, phase=""):
        orig(t, phase)
        st = sim.true_object_state()
        f = sim.contact_forces()
        q = st["quat"]
        tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
        rec.append((sim.time, st["pos"][0], st["pos"][1], st["yaw"], st["lin_vel"][0], st["lin_vel"][1],
                    st["ang_vel"][2], tilt, f["hand"], f["robot"] + f["other"]))

    r._tick = tick
    events = []
    r.hold(0.3, "observe")
    events += r.drain_events()
    samples, labels, actions, pushes = [], [], [], []
    for k in range(K_MAX):
        r.hold(0.3, "settle")
        events += r.drain_events()
        t_dec = r.port.time
        cols, ent_ids, _ = D.events_to_columnar(events, entity_order=[g1.OBJ, g1.PUSHER])
        s = D.tensorize_sample(cols, t_dec, 0, D.MAX_EVENTS)
        x0, y0, yaw0 = (float(v) for v in s["est"])
        pose = (x0, y0, math.cos(yaw0), math.sin(yaw0))
        size = [e for e in events if e.entity_id == g1.OBJ and e.property_type == Prop.SIZE][-1].value
        half = tuple(float(v) for v in size[:3])
        cands = C.generate_candidates(pose, dict(target_xy=ep.target), w.capability, rng, n=48, half=half,
                                      target_entity_id=g1.OBJ, fast_frac=C.SLIDE_TO_TARGET.fast_frac,
                                      max_stroke=C.SLIDE_TO_TARGET.max_stroke)
        feas = [c for c in cands if c.feasible]
        rng.shuffle(feas)
        chosen = plan = None
        for c in feas[:24]:
            p = r.plan_push(c, pose, half)
            if p["feasible"]:
                chosen, plan = c, p
                break
        if chosen is None:
            break
        info = r.execute_push(chosen, pose, half, plan)
        t_push, t_end = info["t_push"], info["t_push_end"]
        while r.port.time < t_push + HOLD_AFTER_PUSH:
            r.hold(0.05, "hold")
        events += r.drain_events()
        samples.append(s)
        actions.append(chosen.vector())
        pushes.append(dict(k=k, t_dec=t_dec, t_push=t_push, t_push_end=t_end, params=list(chosen.params),
                           side=plan["side"], lead=plan["lead"]))
        labels.append(make_labels(np.array(rec), t_push, t_end, (x0, y0, yaw0), HORIZONS, ep))
        st = w.true_object()
        if not (g1.TABLE_XY[0] - g1.TABLE_HALF[0] < st["pos"][0] < g1.TABLE_XY[0] + g1.TABLE_HALF[0]
                and abs(st["pos"][1]) < g1.TABLE_HALF[1]) or st["pos"][2] < g1.TABLE_TOP:
            break  # box left the table
    a = w.audit()
    clean = a["external_object_writes"] == 0 and a["eq_active"] == 0 and a["applied_force_steps"] == 0
    return dict(seed=seed, root=ROOT_BASE + seed, samples=samples, labels=labels, actions=actions,
                params=dict(episode=ep.to_json(), pushes=pushes, audit_clean=bool(clean), sim_time=r.port.time,
                            wall=time.time() - t0))


def make_labels(rec, tk, tend, est, horizons, ep):
    from physmem import data as D

    x0, y0, yaw0 = est
    T = rec[:, 0]
    H = len(horizons)
    idx = np.array([int(np.argmin(np.abs(T - (tk + h)))) for h in horizons])
    fmask = np.array([abs(T[i] - (tk + h)) < 0.006 for i, h in zip(idx, horizons)])
    pose = rec[idx, 1:4]
    vel = rec[idx, 4:7]
    dxy = D.world_to_canonical_xy(pose[:, :2], (x0, y0), yaw0)
    dyaw = pose[:, 2] - yaw0
    v = D.world_to_canonical_vec(vel[:, :2], yaw0)
    fut = np.concatenate([dxy, np.cos(dyaw)[:, None], np.sin(dyaw)[:, None], v, vel[:, 2:3]], 1).astype(np.float32)
    sp = np.hypot(vel[:, 0], vel[:, 1])
    tilt = rec[idx, 7]
    fpush = rec[idx, 8]
    cm = np.where(fpush > D.FORCE_EPS, 1, np.where(tilt > 0.2, 0, np.where((sp > D.SPEED_EPS) | (np.abs(vel[:, 2]) > D.WZ_EPS), 2, 3)))
    fs = np.zeros(H, bool)
    tf = tend + 0.05
    i_f = int(np.argmin(np.abs(T - tf)))
    if T[i_f] >= tf - 0.006 and math.hypot(rec[i_f, 4], rec[i_f, 5]) > D.SPEED_EPS:
        for hi, (h, ii) in enumerate(zip(horizons, idx)):
            if tk + h < tf - 1e-9 or not fmask[hi]:
                continue
            seg = slice(i_f, ii + 1)
            if rec[seg, 8].max() <= D.FORCE_EPS and rec[seg, 9].max() <= D.FORCE_EPS and rec[seg, 7].max() < D.TILT_MAX:
                fs[hi] = True
    return dict(future=fut, future_mask=fmask, contact_mode=cm.astype(np.int64), free_slide_mask=fs,
                friction=np.float32(ep.box.friction), mass=np.float32(ep.box.mass), law=np.int64(0))


def write_split(out_dir, split, roots):
    import torch
    from physmem.schema import EntityType

    smp = [s for r in roots for s in r["samples"]]
    labs = [l for r in roots for l in r["labels"]]
    nk = [len(r["samples"]) for r in roots]
    et = np.array([int(EntityType.OBJECT), int(EntityType.PUSHER), -1, -1], np.int64)
    t = dict(
        entity=torch.from_numpy(np.stack([s["ent"] for s in smp]).astype(np.int8)),
        related=torch.from_numpy(np.stack([s["rel"] for s in smp]).astype(np.int8)),
        prop=torch.from_numpy(np.stack([s["prop"] for s in smp]).astype(np.int8)),
        value=torch.from_numpy(np.stack([s["val"] for s in smp])),
        logvar=torch.from_numpy(np.stack([s["logvar"] for s in smp])).half(),
        age=torch.from_numpy(np.stack([s["age"] for s in smp])),
        latency=torch.from_numpy(np.stack([s["latency"] for s in smp])),
        source=torch.from_numpy(np.stack([s["src"] for s in smp]).astype(np.int8)),
        provenance=torch.from_numpy(np.stack([s["prov"] for s in smp]).astype(np.int8)),
        mask=torch.from_numpy(np.stack([s["mask"] for s in smp])),
        ent_type=torch.from_numpy(np.tile(et, (len(smp), 1))),
        dt_obs=torch.full((len(smp),), 0.05, dtype=torch.float32),
        action=torch.from_numpy(np.stack([a for r in roots for a in r["actions"]]).astype(np.float32)),
        future=torch.from_numpy(np.stack([l["future"] for l in labs])),
        future_mask=torch.from_numpy(np.stack([l["future_mask"] for l in labs])),
        friction=torch.tensor([float(l["friction"]) for l in labs]),
        mass=torch.tensor([float(l["mass"]) for l in labs]),
        contact_mode=torch.from_numpy(np.stack([l["contact_mode"] for l in labs])),
        free_slide_mask=torch.from_numpy(np.stack([l["free_slide_mask"] for l in labs])),
        law=torch.tensor([int(l["law"]) for l in labs]),
        root=torch.from_numpy(np.repeat(np.array([r["root"] for r in roots], np.int64), nk)),
        k=torch.from_numpy(np.concatenate([np.arange(n) for n in nk]).astype(np.int64)),
        est=torch.from_numpy(np.stack([s["est"] for s in smp])),
    )
    torch.save(t, os.path.join(out_dir, f"{split}_tensors.pt"))
    with open(os.path.join(out_dir, f"{split}_params.jsonl"), "w") as f:
        for r in roots:
            f.write(json.dumps(dict(root=r["root"], seed=r["seed"], params=r["params"])) + "\n")
    return t


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/dynagraph/data/g1cal"))
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    s0 = a.seed0
    splits = dict(train=range(s0, s0 + a.n_train), val=range(s0 + a.n_train, s0 + a.n_train + a.n_val),
                  test=range(s0 + a.n_train + a.n_val, s0 + a.n_train + a.n_val + a.n_test))
    man = dict(dataset="g1cal", experiment="e15", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               generator="scripts/demo_g1_make_g1cal.py", controller="g1 push primitive v1 (PUSH_SERVO=v1)",
               candidate_distribution="control.generate_candidates, SLIDE_TO_TARGET (fast_frac 0.4, stroke<=0.08 m, "
                                      "dur 0.1-0.5 s, speed log-U[0.05,0.8] + fast U[0.45,0.8]); uniform choice among "
                                      "IK-feasible", hidden=dict(friction=[0.3, 0.7], mass_loguniform=[0.2, 2.0]),
               k_max=K_MAX, hold_after_push_s=HOLD_AFTER_PUSH, root_id="9000000 + seed",
               seeds={k: [v.start, v.stop - 1] for k, v in splits.items()},
               disjoint_from="G1 test seeds 0-49 (slide/easy), servo-calibration seeds 10000-10049",
               scales="copied byte-identical from data/v2/scales.json (NOT refit)", splits={})
    with Pool(a.procs) as pool:
        for name, seeds in splits.items():
            ts = time.time()
            roots = pool.map(gen_root, list(seeds), chunksize=2)
            roots = [r for r in roots if r["samples"]]
            t = write_split(a.out, name, roots)
            n_s = int(len(t["action"]))
            man["splits"][name] = dict(n_roots=len(roots), n_samples=n_s,
                                       k_hist=np.bincount(t["k"].numpy(), minlength=K_MAX).tolist(),
                                       audit_clean=all(r["params"]["audit_clean"] for r in roots),
                                       free_slide_frac=float(t["free_slide_mask"].float().mean()),
                                       future_mask_frac=float(t["future_mask"].float().mean()),
                                       wall_s=round(time.time() - ts, 1),
                                       sha256=sha(os.path.join(a.out, f"{name}_tensors.pt")))
            print(name, json.dumps(man["splits"][name]), flush=True)
    v2s = os.path.expanduser("~/dynagraph/data/v2/scales.json")
    shutil.copyfile(v2s, os.path.join(a.out, "scales.json"))
    man["scales_sha256"] = sha(os.path.join(a.out, "scales.json"))
    man["wall_total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
