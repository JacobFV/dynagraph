"""dev tool: trace robot/gripper state during the first k scripted calls of a family."""
import sys, numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
fam, seed, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1
every = int(sys.argv[4]) if len(sys.argv) > 4 else 100
env = make_env(fam)
env.reset(TaskSpec(fam, env.category, seed, "probe", "train"))
r = env.robot
orig = env.sensors.tick
cnt = [0]
def tick(t):
    orig(t)
    cnt[0] += 1
    if cnt[0] % every == 0:
        tgt = getattr(env, "_trace_target", None)
        extra = ""
        if tgt is not None and tgt in env.body_id:
            extra = f" obj={np.round(env.true_pose(tgt)[0], 3)}"
        print(f"t={t:.2f} tcp={np.round(r.tcp()[0],3)} yaw={r.q()[3]:.2f} ref={np.round(r.ref[:3]+r.mount_pos,3)} grip={r.grip_cmd} w={r.grip_width():.4f} pads={tuple(round(float(x),1) for x in r.pad_contacts())} u={np.round(env.d.ctrl[r.act],1)}{extra}")
env.sensors.tick = tick
for i in range(k):
    c = env.scripted()
    if c is None: break
    env._trace_target = c.target
    print("CALL", c.skill, c.target, {a: round(b, 3) if isinstance(b, float) else b for a, b in c.params.items()})
    if c.target in env.scene.entities: print("   hidden", env.scene.entities[c.target].hidden, "half", env.scene.entities[c.target].half)
    out = env.execute(c)
    print("  ->", out.result.status, out.result.log, "progress", out.metrics["progress"])
