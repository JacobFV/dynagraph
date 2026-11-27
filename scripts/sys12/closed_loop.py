"""Deliverable 6: paired closed loop, System II (Inkling-2-Mini base LM, adapters off) plans the skill sequence, System I
sets the continuous params. Pre-registered in docs/INKLING_SYS12.md section a.
usage: CELL=bin_ur5e|bin_gen3|pouch_ur5e N=30 python scripts/sys12/closed_loop.py <tag>
env: SPLIT (default test; val = smoke only), SEED_OFF (default 950000), S1_CKPT, POLICIES (comma list), EPISODE_TIMEOUT
policies (identical seeded tasks / lots):
  heuristic            plain scripted (never probes)
  pta                  ENVS probe-then-act scripted
  v2                   MODEL-V2 alone: scripted skill choice + MODEL-V2 r7 planner (bin lots: crush-gated LotPlanner
                       'typed_v2'; pouch: residual_gated_probe ResidualPlanner) - the current best system
  s12_native           System II (Inkling) chooses skill+target -> System I = Inkling-2-Mini core (pretrained) in fast
                       mode (gated adapters, 1 core pass) via the MODEL-V2 System1Head scoring rule
  s12_delegated        System II (Inkling) -> System I = MODEL-V2 r7 typed 3-seed ensemble (System1Head, CPU)
  s1native_scripted    scripted skill choice -> System I native (isolates the System II contribution)"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from physmem.envs import make_env  # noqa: E402
from physmem.models.v2 import harness as H  # noqa: E402
from physmem.models.v2 import spec as S  # noqa: E402
from physmem.models.v2.env_adapter import EnvAdapter  # noqa: E402
from physmem.models.v2.lot import LotEnvAdapter, LotPlanner, _probed  # noqa: E402
from physmem.planner import CEMPlanner, PlannerConfig  # noqa: E402
from physmem.v2 import api  # noqa: E402
from physmem.v2_train import SEED_BASE, load_model  # noqa: E402

tag = sys.argv[1]
CELL = os.environ.get("CELL", "bin_ur5e")
N = int(os.environ.get("N", "30"))
SPLIT = os.environ.get("SPLIT", "test")
SEED_OFF = int(os.environ.get("SEED_OFF", "950000"))
S1_CKPT = os.environ.get("S1_CKPT", "results/sys12/checkpoints/s1r1_inkling_gated_s0.pt")
TAU, KAPPA, LAM_T = 0.15, 1.0, 0.01
GRIDS = {"bin_clearing": {"grip_force": [3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0], "lift_speed": [0.2, 0.1, 0.05]},
         "pouch_transfer": {"grip_force": [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0], "lift_speed": [0.2, 0.1, 0.05]}}
fam, robot = {"bin_ur5e": ("bin_clearing", "ur5e_2f85"), "bin_gen3": ("bin_clearing", "gen3_2f85"),
              "pouch_ur5e": ("pouch_transfer", "ur5e_2f85")}[CELL]
split = "heldout_robot" if robot == "gen3_2f85" and SPLIT == "test" else SPLIT
POL = os.environ.get("POLICIES", "heuristic,pta,v2,s12_native,s12_delegated,s1native_scripted").split(",")
C = "results/v2/checkpoints"
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


# ------------------------------------------------------------------------------------------- System II menu
def menu(ad):
    """actor-legal cell state lines + allowed (skill, target) calls; [] -> the cell is done / nothing to do."""
    env = ad.env
    if fam == "bin_clearing":
        if env.held is not None:
            return [f"holding {env.held} in the gripper"], [("place", "tote")]
        pk = env._pickable()
        st = ["gripper empty"]
        if not pk:
            looks = getattr(env, "_looks", 0)
            if looks < 2:
                return st + ["no part currently visible in the source bin"], [("reach", "tote")]
            pk = env._pickable(stale_ok=True)
            if not pk:
                return st, []
        for r, e in enumerate(pk[:4]):
            f = env.fail_count.get(e, 0)
            st.append(f"{e} in the source bin, accessibility rank {r + 1} (1 = most accessible)"
                      + (f", {f} failed grasp(s)" if f else "") + (", already probed" if _probed(env, e) else ""))
        return st, [("pick", e) for e in pk[:4]]
    # pouch
    if env.held is not None:
        return ["holding the pouch"], [("place", "tote")]
    if env.n_calls > 0 and env.success():
        return ["pouch is in the tote"], []
    return ["gripper empty", "pouch on the bench"], [("pick", "pouch")]


def base_call(ad, skill, target, sc):
    """the scripted parameterisation of the chosen (skill, target) (System I then rewrites pick params).
    sc = this decision's env.scripted() call (computed ONCE per decision: it has side effects)."""
    env = ad.env
    if sc is not None and sc.skill == skill and sc.target == target:
        return sc
    if fam == "bin_clearing":
        from physmem.envs.families_bulk import _pick_call
        if skill == "pick":
            k = env.fail_count.get(target, 0)
            return _pick_call(target, grip_force=(40.0, 90.0, 100.0, 100.0)[min(k, 3)],
                              lift_speed=(0.2, 0.1, 0.08, 0.08)[min(k, 3)], depth=(0.5, 0.5, 0.3, 0.7)[min(k, 3)])
        if skill == "place":
            return env._place_call()
        reg = env.scene.regions["tote"]
        return api.SkillCall("reach", dict(x=float(reg["center"][0]), y=float(reg["center"][1]) - 0.3,
                                           z=float(env.safe_z), speed=0.6, dwell=0.6), target="tote")
    if skill == "place":
        return api.SkillCall("place", dict(release_h=0.03, speed=0.3), target="tote")
    return api.SkillCall("pick", dict(grip_force=12.0, lift_speed=0.2, depth=0.0), target="pouch")


class Sys12Policy:
    """System II (router or scripted) -> System I (System1Head over native or delegated planners)."""
    search = "lot"

    def __init__(self, name, head, router=None):
        self.name, self.head, self.router = name, head, router
        self.mode = "typed"
        self.log = []

    def reset(self):
        pass

    def decide(self, ad, hist, emb, rng):
        env = ad.env
        t0 = time.perf_counter()
        st, opts = menu(ad)
        if not opts:
            return None, {}
        sc = env.scripted()
        if self.router is None:
            if sc is None:
                return None, {}
            choice, base, t_s2 = (sc.skill, sc.target), sc, 0.0
        else:
            from physmem.models.sys12 import FAMILY_TEXT, format_handoff, parse_handoff, sys2_messages
            if len(opts) == 1:                            # forced: a single allowed call needs no deliberation
                j = 0
            else:
                j, _ = self.router.choose(sys2_messages(FAMILY_TEXT[fam], st, opts), opts)
            text = format_handoff(*opts[j])               # the System II -> System I handoff line
            choice = parse_handoff(text)
            base = base_call(ad, *choice, sc)
            t_s2 = time.perf_counter() - t0
        info = dict(kind="scripted", sys2=list(choice), sys2_ms=1000 * t_s2, n_opts=len(opts),
                    scripted=[getattr(sc, "skill", None), getattr(sc, "target", None)])
        self.log.append(info)
        is_part = base.skill == "pick" and "grip_force" in base.params and (
            base.target in getattr(ad, "lot_ids", set()) or fam == "pouch_transfer")
        if not is_part:
            return base, info
        t1 = time.perf_counter()
        for pl in self.head.planners:
            pl.skill = "pick"
        q = ad._tid(ad.target_ids(base.target), base.target)
        out = self.head.propose(base, hist, ad.frame(base.target), ad.embodiment_spec(), ad.target_pos(base.target),
                                query=q, grids=[GRIDS[fam]], rng=np.random.default_rng(0))
        best = out["candidates"][0]
        info.update(sys1_ms=1000 * (time.perf_counter() - t1), ucb=best["ucb"], risk=best["score"],
                    grip=float(out["call"].params.get("grip_force", np.nan)),
                    lift=float(out["call"].params.get("lift_speed", np.nan)))
        if best["ucb"] > TAU and not _probed(env, base.target):
            info["kind"] = "probe"
            return env.probe_then_act(base), info
        info["kind"] = "skip"
        return out["call"], info


# ------------------------------------------------------------------------------------------- build policies
pols = {}
need_lm = any(p in POL for p in ("s12_native", "s12_delegated", "s1native_scripted"))
if "heuristic" in POL:
    pols["heuristic"] = ("heuristic", None)
if "pta" in POL:
    pols["pta"] = ("heuristic_probe", None)
from physmem.models.v2.system1 import System1Head  # noqa: E402

if "v2" in POL:
    if fam == "bin_clearing":
        T = [CEMPlanner(load_model(f"{C}/v2_robots_r7_typed_s{s}.pt", torch.device("cpu"))[0], "typed",
                        PlannerConfig(n_z=16, voi=False)) for s in (0, 1, 2)]
        pols["v2"] = ("planner", LotPlanner(T, tau=TAU, name="typed_v2", crush_gate=True))
    else:
        cfg = PlannerConfig(pop=96, iters=3, n_z=16, voi=False, c_prog=2.0, lam_effort=0.03)
        T = [CEMPlanner(load_model(f"{C}/v2_robots_r7_typed_s{s}.pt", torch.device("cpu"))[0], "typed", cfg)
             for s in (0, 1, 2)]
        tr = {f: v["trusted"] for f, v in json.load(open("results/v2/trust_r6f.json"))["families"].items()}
        rp = H.ResidualPlanner(T, trust=tr)
        rp.anchor_probe = True
        pols["v2"] = ("planner", rp)
lm_info = {}
if need_lm:
    from physmem.models.sys12 import Sys2Router, sys1_planner
    from physmem.sys12_train import build, load_state
    ck = torch.load(S1_CKPT, map_location="cpu", weights_only=False)
    m, rep = build(ck["info"]["arm"], ck["info"]["passes"], ck["info"]["seed"])
    load_state(m, ck["state"])
    m.eval()
    m.family = fam                       # System I context = this task family's System II text
    from physmem.sys12_train import _LM
    model, tm, tok, _ = next(iter(_LM.values()))
    router = Sys2Router(model, tm, tok, core=m.core)
    native = System1Head([sys1_planner(m)], lam_t=LAM_T, kappa=KAPPA)
    lm_info = dict(s1_ckpt=S1_CKPT, s1_info=ck["info"])
    if "s12_native" in POL:
        pols["s12_native"] = ("planner", Sys12Policy("s12_native", native, router))
    if "s1native_scripted" in POL:
        pols["s1native_scripted"] = ("planner", Sys12Policy("s1native_scripted", native, None))
    if "s12_delegated" in POL:
        deleg = System1Head.load(f"{C}/v2_robots_r7_typed_s{{}}.pt", seeds=(0, 1, 2), device="cpu",
                                 lam_t=LAM_T, kappa=KAPPA)
        pols["s12_delegated"] = ("planner", Sys12Policy("s12_delegated", deleg, router))
pols = {k: v for k, v in pols.items() if k in POL}

# ------------------------------------------------------------------------------------------- run (resumable)
params = {"dressing": "site"}
if fam == "bin_clearing":
    params["n_parts"] = 3
factory = (lambda: LotEnvAdapter(make_env(fam))) if fam == "bin_clearing" else (lambda: EnvAdapter(make_env(fam)))
tasks = [api.TaskSpec(fam, "", SEED_BASE.get(split, 2_000_000) + SEED_OFF + i, robot, split, params=dict(params))
         for i in range(N)]
out_dir = "results/sys12/closed_loop"
os.makedirs(out_dir, exist_ok=True)
pp = f"{out_dir}/{tag}_{CELL}.partial.json"
res, start = {k: [] for k in pols}, 0
if os.path.exists(pp):
    old = json.load(open(pp))
    if set(old["episodes"]) == set(pols):
        res, start = old["episodes"], int(old["done_tasks"])
        print(f"resume at task {start}", flush=True)
t0 = time.time()
for c0 in range(start, N, 5):
    part = H.paired_eval(factory, tasks[c0:c0 + 5], pols, offset=c0)
    with open(f"{out_dir}/{tag}_{CELL}.declog.jsonl", "a") as f:
        for k, (pol, pl) in pols.items():
            if isinstance(pl, Sys12Policy):
                for d in pl.log:
                    f.write(json.dumps(dict(policy=k, chunk=c0, **d), default=float) + "\n")
                pl.log = []
    for k in res:
        res[k].extend(part[k])
    json.dump(dict(episodes=res, done_tasks=c0 + len(tasks[c0:c0 + 5]), cell=CELL, family=fam, robot=robot, split=split,
                   seed_off=SEED_OFF, tau=TAU, lm=lm_info), open(pp, "w"), default=float)
    print(f"[{CELL}] tasks {c0 + 5}/{N} ({time.time() - t0:.0f}s): " + " ".join(
        f"{k}={np.nanmean([e.get('good_frac', np.nan) if fam == 'bin_clearing' else float(e['success']) for e in v]):.2f}"
        for k, v in res.items()), flush=True)
json.dump(dict(episodes=res, cell=CELL, family=fam, robot=robot, split=split, seed_off=SEED_OFF, tau=TAU, lm=lm_info,
               seconds=time.time() - t0), open(f"{out_dir}/{tag}_{CELL}.json", "w"), default=float)
print("DONE", flush=True)
