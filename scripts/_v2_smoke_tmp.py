import numpy as np, time
from physmem.models.v2 import synth as Y, spec as S
for fam in Y.FAMILIES:
    t=time.time(); succ=[]; ev=[]
    for s in range(200):
        ep=Y.generate_episode(fam,"synth_arm_a",s)
        for st in ep.steps:
            it=st["inter"]; assert it.segment.shape==(64,26) and np.isfinite(it.segment).all(), fam
            ev.append(it.outcome_events); succ.append(it.outcome_cont)
            f=S.analytic_features(it.segment); assert np.isfinite(f).all()
    ev=np.array(ev); c=np.array(succ)
    print(fam, f"{(time.time()-t)*1000/len(ev):.2f}ms/step", "event rates", dict(zip(S.OUT_EVENTS, ev.mean(0).round(3))), "cont mean", c.mean(0).round(3))
