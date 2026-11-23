"""MODEL-V2: inspect one ENVS StepOut to build the event->segment adapter."""
import sys, collections, numpy as np
from physmem.envs import make_env, FAMILIES
from physmem.v2.api import TaskSpec
print("families", FAMILIES)
fam = sys.argv[1] if len(sys.argv) > 1 else "bulk_sweep"
env = make_env(fam)
ev = env.reset(TaskSpec(fam, "bulk", seed=1, robot="probe", split="train", params={}))
print("reset events", len(ev), collections.Counter(str(e.property_type) for e in ev).most_common())
print("entities", sorted({e.entity_id for e in ev})[:30])
rng = np.random.default_rng(0)
c = env.candidate_skills(None, 4, rng)
for x in c: print("cand", x.skill, x.target, {k: (v if np.isscalar(v) else type(v).__name__) for k, v in x.params.items()})
print("scripted", env.scripted())
out = env.execute(c[0])
print("step events", len(out.events), collections.Counter(str(e.property_type) for e in out.events).most_common())
print("result", out.result.ok, out.result.status, out.result.duration_s, list(out.result.log.keys()))
print("metrics", out.metrics)
print("labels keys", list(out.labels.keys()), "hidden sample", list(out.labels["hidden"].items())[:3])
print("goal", {k: (v if not isinstance(v, dict) else list(v)[:5]) for k, v in env.goal().items()})
print("robot spec", getattr(env.robot, "spec", None))
