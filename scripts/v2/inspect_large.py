import sys, glob, collections, numpy as np
from physmem.envs.io import iter_episodes
pat = sys.argv[1]
for i, ep in enumerate(iter_episodes(pat)):
    print(type(ep), list(ep.keys()) if isinstance(ep, dict) else dir(ep)[:40])
    for k, v in ep.items():
        if k != "steps" and k != "events":
            print(" ", k, str(v)[:300])
    print("n events", len(ep.get("events", [])), "n steps", len(ep["steps"]))
    st = ep["steps"][0]
    for k, v in st.items():
        print("  step.", k, str(v)[:400])
    break
