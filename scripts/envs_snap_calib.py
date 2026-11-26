"""r7 dev: tight_insert snap-fit calibration - which press force passes the lips for a given lip stiffness / friction?
    python scripts/envs_snap_calib.py <robot> <part> <nparts>"""
import itertools, json, sys, warnings
warnings.filterwarnings("ignore")
from physmem.envs import make_env
from physmem.envs.splits import split_params
from physmem.v2.api import TaskSpec, SkillCall
rob, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
grid = list(itertools.product((5000.0, 12000.0, 25000.0), (0.3, 0.7), (4.0, 8.0, 13.0, 20.0, 28.0)))[part::nparts]
for k, mu, F in grid:
    env = make_env("tight_insert")
    env.reset(TaskSpec("tight_insert", "fine", 1700001, rob, "train",
                       dict(split_params("tight_insert", "train"), snap_k=k, mu=mu, dent_n=float("inf"), clearance=0.0003,
                            chamfer=0.0004)))
    st = []
    for _ in range(4):
        c = env.scripted()
        if c is None:
            break
        if c.skill == "insert":
            c = SkillCall("insert", dict(c.params, down_force=F, min_force=0.0, wiggle=0.0, retries=0), target=c.target)
        o = env.execute(c)
        st.append(o.result.status)
        if o.done or c.skill == "insert":
            break
    print(json.dumps(dict(k=k, mu=mu, F=F, success=bool(env.solved()), depth=round(env.depth_frac(), 2), status=st)), flush=True)
