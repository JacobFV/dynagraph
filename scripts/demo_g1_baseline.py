"""Gate artefact: one G1 hand push on the dynamic box (no learned model involved).
Saves a before/after screenshot + config + measured box displacement to results/g1/baseline/."""
import json, math, sys, time
from pathlib import Path
import mujoco, numpy as np
from PIL import Image
from physmem.robots import g1
from physmem.schema import ActionCandidate

out = Path("results/g1/baseline"); out.mkdir(parents=True, exist_ok=True)
ep = g1.sample_push_episode(3, friction=0.5, mass=0.6); ep.box.yaw = 0.3; ep.box.pos = (0.39, -0.05)
w = g1.G1PushWorld(ep); r = w.actor; sim = w._sim
ren = mujoco.Renderer(sim.model, 720, 1280)
def shot(name):
    ren.update_scene(sim.data, camera="overview"); Image.fromarray(ren.render()).save(out / name)
cand = ActionCandidate("baseline", "push", (math.pi, 0.0, 0.3, 0.4), g1.OBJ)
st0 = w.true_object(); shot("before.png")
pose = (st0["pos"][0], st0["pos"][1], math.cos(st0["yaw"]), math.sin(st0["yaw"]))
frames = []
orig = r._tick
def tick(t, phase=""):
    orig(t, phase)
    if phase == "push" and not frames and sim.contact_forces()["hand"] > 0.5:
        shot("contact.png"); frames.append(sim.time)
r._tick = tick
info = r.execute_push(cand, pose, ep.box.half); r.hold(1.0)
st1 = w.true_object(); shot("after.png")
rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), episode=ep.to_json(), candidate=cand.params,
           push=info, box_start=st0["pos"].tolist(), box_end=st1["pos"].tolist(),
           displacement_m=float(np.linalg.norm(st1["pos"][:2] - st0["pos"][:2])), dyaw=st1["yaw"] - st0["yaw"],
           max_hand_force=max(x["f_hand"] for x in w.trace()), audit=w.audit(), contact_shot_t=frames,
           config=dict(TABLE_TOP=g1.TABLE_TOP, PUSH_Z=g1.PUSH_Z, PUSHER_SITE_X=g1.PUSHER_SITE_X, PUSH_LEADS=g1.PUSH_LEADS,
                       ACTUATOR_TUNING=g1.ACTUATOR_TUNING, PHYSICS_DT=g1.PHYSICS_DT, CONTROL_DT=g1.CONTROL_DT,
                       HAND_FRICTION=g1.HAND_FRICTION, menagerie_commit=g1.MENAGERIE_COMMIT))
(out / "baseline.json").write_text(json.dumps(rec, indent=1, default=float))
print(json.dumps({k: rec[k] for k in ("utc", "displacement_m", "dyaw", "max_hand_force", "audit")}, default=float))
