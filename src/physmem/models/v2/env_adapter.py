"""Adapter: ENVS StationEnv (api.IndustrialEnv) event streams -> v2 model records (owner: MODEL-V2).

ENVS emits per-tick actor-legal ObservationEvents (envs/props.XProp: WRENCH, EE_POSE, GRIPPER, TACTILE, POSE3, JOINT,
REGION, SIZE) instead of calling spec.SegmentRecorder, so segments are rebuilt here from the events whose
measured_at falls inside each skill's execution window. Commanded-trajectory channels are zero unless the skill
result log carries them (documented; the model sees the same convention in training and at decision time).

Skill parameters that are world coordinates (keys x, y, z, x0, y1, *_x, ...) are made relative to the target's
actor-estimated position before vectorizing (translation equivariance), and converted back when executing.
"""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from physmem.v2 import api
from . import spec as S

STIFF_PROP = 7   # schema.Prop.STIFFNESS (ENVS r6 vision_material / tactile_squeeze streams)
POSKEY = re.compile(r"^([xyz])\d*$|^.*_([xyz])$")
AX = {"x": 0, "y": 1, "z": 2}
# ENVS hidden-label names -> spec.LATENTS
NOT_FRAGILE_N = 500.0    # crush/dent label used for parts ENVS marks unbreakable (crush_n = inf)
HIDDEN_MAP = {"mass": "mass", "mu": "friction", "mu_grip": "grip_friction", "clearance": "clearance",
              "damping": "damping", "frictionloss": "breakaway", "crush_n": "crush", "dent_n": "crush",
              "stiffness": "stiffness"}

try:  # ENVS extended props (absent -> adapter unusable but importable)
    from physmem.envs.props import XProp
except Exception:  # pragma: no cover
    XProp = None


def _axis(k):
    m = POSKEY.match(k)
    return AX[m.group(1) or m.group(2)] if m else None


def rel_params(params: dict, p0) -> dict:
    return {k: (float(v) - float(p0[_axis(k)]) if (_axis(k) is not None and np.isscalar(v)) else v)
            for k, v in params.items()}


def abs_params(params: dict, p0) -> dict:
    return {k: (float(v) + float(p0[_axis(k)]) if (_axis(k) is not None and np.isscalar(v)) else v)
            for k, v in params.items()}


class EventIndex:
    """per (entity, prop) time-sorted arrays of (measured_at, available_at, value)."""

    def __init__(self):
        self.d = {}

    def add(self, events):
        for e in events:
            self.d.setdefault((e.entity_id, int(e.property_type)), []).append(
                (e.measured_at, e.available_at, np.asarray(e.value, float)))
            if int(e.property_type) == STIFF_PROP:      # r6 fragility evidence, kept per source
                self.d.setdefault((e.entity_id, int(e.property_type), e.source_id), []).append(
                    (e.measured_at, e.available_at, np.asarray(e.value, float)))

    def latest_src(self, eid, prop, src, now):
        rows = [r for r in self.d.get((eid, int(prop), src), []) if r[1] <= now]
        return max(rows, key=lambda r: r[0])[2] if rows else None

    def series(self, eid, prop, t0=-np.inf, t1=np.inf):
        rows = sorted(r for r in self.d.get((eid, int(prop)), []) if t0 <= r[0] <= t1)
        if not rows:
            return np.zeros(0), np.zeros((0, 1))
        return np.array([r[0] for r in rows]), np.stack([r[2] for r in rows])

    def latest(self, eid, prop, now):
        rows = [r for r in self.d.get((eid, int(prop)), []) if r[1] <= now]
        return max(rows, key=lambda r: r[0])[2] if rows else None

    def entities(self, prop):
        return sorted({k[0] for k in self.d if len(k) == 2 and k[1] == int(prop)})


def build_segment(idx: EventIndex, t0, t1, target_ids, p0, hand="robot/hand", max_rows=400):
    """events in [t0,t1] -> (T_raw, N_CH) segment (positions relative to p0)."""
    if not (t1 - t0 > 1e-6):            # r2-r5 zero-length (rejected) calls: give the window a nominal length
        t1 = t0 + 0.05
    tw, W = idx.series(hand, XProp.WRENCH, t0, t1)
    if len(tw) < 2:
        tw, _ = idx.series(hand, XProp.EE_POSE, t0, t1)
    tw = np.unique(tw) if len(tw) else tw
    if len(tw) < 2:
        tw = np.linspace(t0, t1, 8)
    if len(tw) > max_rows:
        tw = tw[np.linspace(0, len(tw) - 1, max_rows).astype(int)]
    x = np.zeros((len(tw), S.N_CH), np.float32)

    def interp(eid, prop, cols):
        t, v = idx.series(eid, prop, t0 - 0.2, t1 + 0.05)
        if len(t) == 0:
            return np.zeros((len(tw), len(cols)))
        return np.stack([np.interp(tw, t, v[:, c]) for c in cols], 1)

    ee = interp(hand, XProp.EE_POSE, (0, 1, 2)) - p0[None]
    x[:, 6:9] = ee
    x[:, 9:12] = np.nan_to_num(np.gradient(ee, tw, axis=0)) if len(tw) > 1 else 0
    x[:, 12:18] = interp(hand, XProp.WRENCH, range(6))
    g = interp(hand, XProp.GRIPPER, (0, 1, 2))
    x[:, S.CH["grip_pos"]] = g[:, 0]
    x[:, S.CH["grip_cmd"]] = np.clip(np.maximum(g[:, 1] / 50.0, g[:, 2]), 0, 1)
    tac = interp(hand, XProp.TACTILE, (0, 1, 2))
    x[:, S.CH["contact"]] = (tac.max(1) > 0.5).astype(np.float32)
    # target observation: mean POSE3 position of target ids, fresh flag when a new sample was measured in the bin
    obs_t, obs_v = [], []
    for eid in target_ids:
        t, v = idx.series(eid, XProp.POSE3, t0, t1)
        if len(t) == 0:      # articulated target observed through its joint coordinate (drawer / door)
            t, v = idx.series(eid, XProp.JOINT, t0, t1)
            v = np.concatenate([v[:, :1], np.zeros((len(v), 2))], 1) if len(v) else np.zeros((0, 3))
            obs_t.append(t)
            obs_v.append(v + p0[None] if len(v) else v)
            continue
        obs_t.append(t)
        obs_v.append(v[:, :3] if len(v) else np.zeros((0, 3)))
    held = np.zeros(3)
    prev = -np.inf
    allt = np.concatenate(obs_t) if obs_t else np.zeros(0)
    for i, t in enumerate(tw):
        fresh = ((allt > prev) & (allt <= t)).any() if len(allt) else False
        if fresh:
            vals = [v[(tt > prev) & (tt <= t)].mean(0) for tt, v in zip(obs_t, obs_v) if ((tt > prev) & (tt <= t)).any()]
            held = np.mean(vals, 0) - p0
        x[i, 20:23] = held
        x[i, 23] = float(fresh)
        prev = t
    x[:, S.CH["t"]] = tw - tw[0]
    return x


class EnvAdapter:
    """wraps an ENVS StationEnv for the v2 harness/planner/data collection.
    hooks used by harness.run_episode: reset, frame, execute, param_bounds, probe_skills, heuristic_params,
    embodiment_spec, lat (evaluator labels as spec latents for the oracle / latent tracking)."""

    def __init__(self, env):
        self.env = env
        self.task = None

    # ---------------------------------------------------------- lifecycle
    def reset(self, task):
        self.task = task
        self.idx = EventIndex()
        ev = self.env.reset(task)
        self.idx.add(ev)
        self.k = 0
        self._labels = None
        self.lat = {}
        self.pool = self.env.candidate_skills(None, 64, np.random.default_rng(task.seed))
        self._damaged_prev = 0
        return ev

    @property
    def now(self):
        return float(self.env.t)

    def embodiment_spec(self):
        return getattr(self.env.robot, "spec", None)

    # ---------------------------------------------------------- actor-legal state
    def object_ids(self):
        return [e for e in self.idx.entities(XProp.POSE3)]

    def target_ids(self, target):
        objs = self.object_ids()
        if target in objs or target in self.idx.entities(XProp.JOINT):
            return [target]
        held = getattr(self.env, "held", None)      # place/transport skills act on the held object
        if held in objs:
            return [held]
        return objs          # bulk / region targets -> the whole pile

    def target_pos(self, target):
        ps = [self.idx.latest(e, XProp.POSE3, self.now) for e in self.target_ids(target)]
        ps = [p[:3] for p in ps if p is not None]
        if not ps and target is not None:      # articulated: use the env's actor-legal handle estimate if any
            reg = self.idx.latest(target, XProp.REGION, self.now)
            if reg is not None:
                ps = [np.asarray(reg[:3])]
        return np.mean(ps, 0) if ps else np.zeros(3)

    def frame(self, target=None) -> S.EntityFrame:
        """actor-legal entity snapshot at `now`, positions relative to the target; r6: per-object stiffness evidence
        (latest vision-material and tactile-squeeze readings)."""
        target = target or (self.pool[0].target if self.pool else None)
        p0 = self.target_pos(target)
        ET = S.ENT_TYPES.index
        bulk = len(self.object_ids()) > 12
        NAN2 = (np.nan, np.nan)

        def stiff_of(e):
            vk = self.idx.latest_src(e, STIFF_PROP, "vision_material", self.now)
            tk = self.idx.latest_src(e, STIFF_PROP, "tactile_squeeze", self.now)
            return (float(vk[0]) if vk is not None else np.nan, float(tk[0]) if tk is not None else np.nan)

        rows = []      # (type, pos, size, stiff(2))
        tids = self.target_ids(target)
        if len(tids) > 1:       # pooled pile entity as the target token
            ps = np.array([self.idx.latest(e, XProp.POSE3, self.now)[:3] for e in tids
                           if self.idx.latest(e, XProp.POSE3, self.now) is not None])
            rows.append((ET("object"), (ps.mean(0) if len(ps) else np.zeros(3)) - p0,
                         ps.std(0) * 2 if len(ps) else np.full(3, 0.01), NAN2))
        for e in self.object_ids():
            v = self.idx.latest(e, XProp.POSE3, self.now)
            if v is None:
                continue
            sz = self.idx.latest(e, S_PROP_SIZE, self.now)
            sz = np.asarray(sz[:3]) if sz is not None else np.full(3, 0.02)
            r = (ET("particle") if bulk else ET("object"), v[:3] - p0, sz, stiff_of(e))
            if len(tids) == 1 and e == tids[0]:
                rows.insert(0, r)
            else:
                rows.append(r)
        for e in self.idx.entities(XProp.REGION):
            v = self.idx.latest(e, XProp.REGION, self.now)
            if v is None:
                continue
            name = e.lower()
            typ = ET("goal_region") if any(k in name for k in ("target", "tote", "goal", "kit", "place", "hole")) \
                else ET("fixture")
            rows.append((typ, np.asarray(v[:3]) - p0, np.asarray(v[3:6]), NAN2))
        ee = self.idx.latest("robot/hand", XProp.EE_POSE, self.now)
        if ee is not None:
            rows.append((ET("effector"), ee[:3] - p0, np.full(3, 0.02), NAN2))
        for e in self.idx.entities(XProp.JOINT):
            if e.startswith("robot"):
                continue
            v = self.idx.latest(e, XProp.JOINT, self.now)
            rows.append((ET("link"), np.array([v[0], 0, 0]), np.full(3, 0.02), NAN2))
        if not rows:
            rows = [(ET("object"), np.zeros(3), np.full(3, 0.02), NAN2)]
        n = len(rows)
        return S.EntityFrame(types=np.array([r[0] for r in rows]), pos=np.array([r[1] for r in rows], float),
                             quat=np.tile([1.0, 0, 0, 0], (n, 1)), size=np.array([r[2] for r in rows], float),
                             vel=np.zeros((n, 3)), age=np.zeros(n), valid=np.ones(n, bool), target=0,
                             stiff=np.array([r[3] for r in rows], float))

    # ---------------------------------------------------------- skills
    def candidate_skills(self, belief, n, rng):
        return self.env.candidate_skills(belief, n, rng)

    def param_bounds(self, skill=None):
        """scalar-param bounds (target-relative for position keys) from the env's own candidate sampler."""
        pool = [c for c in self.pool if skill is None or c.skill == skill] or self.pool
        p0 = self.target_pos(pool[0].target)
        keys = sorted(k for k, v in pool[0].params.items() if np.isscalar(v))
        vals = np.array([[rel_params(c.params, p0)[k] for k in keys] for c in pool])
        lo, hi = vals.min(0), vals.max(0)
        return {k: (float(a), float(b) if b > a else float(a) + 1e-6) for k, a, b in zip(keys, lo, hi)}

    def skill_of_pool(self):
        return self.pool[0].skill if self.pool else None

    def heuristic_params(self):
        c = self.env.scripted()
        return None if c is None else rel_params(c.params, self.target_pos(c.target))

    def probe_skills(self, belief=None):
        if hasattr(self.env, "probe_skills"):
            return self.env.probe_skills(belief)
        return []

    def call_vector(self, call: api.SkillCall):
        p0 = self.target_pos(call.target)
        return replace(call, params=rel_params(call.params, p0)).vector()

    def oracle_latents(self, target):
        """EVALUATOR-ONLY (labelled oracle planner): true latents of the target from env labels."""
        return map_labels(self.env.labels(), self.target_ids(target))

    def execute(self, call: api.SkillCall, relative=True, base=None) -> api.StepOut:
        """relative=True: scalar position params are target-relative (planner output) -> converted to world.
        non-scalar params are taken from `base` (the env candidate the plan was seeded from)."""
        target = call.target or (self.pool[0].target if self.pool else None)
        p0 = self.target_pos(target)
        base = base or next((c for c in self.pool if c.skill == call.skill), None)
        params = dict(base.params) if base is not None else {}
        params.update(abs_params(call.params, p0) if relative else call.params)
        wcall = replace(call, params=params, target=target)
        tids = self.target_ids(target)          # resolve BEFORE executing (a place releases the held object)
        t0 = self.now
        before = dict(self.env.metrics_acc) if hasattr(self.env, "metrics_acc") else {}
        before["damaged"] = getattr(self, "_damaged_prev", 0)
        out = self.env.execute(wcall)
        self.idx.add(out.events)
        t1 = self.now
        seg = build_segment(self.idx, t0, t1, tids, p0)
        p1 = self.target_pos(target)
        rcall = replace(wcall, params=rel_params(wcall.params, p0))
        m = out.metrics
        self._damaged_prev = _num(m.get("damaged", 0))
        peak = float(np.abs(seg[:, 12:15]).max()) if len(seg) else 0.0
        cont = np.array([*(p1 - p0), float(m.get("progress", 0.0)), peak / 50.0, out.result.duration_s,
                         float(m.get("progress_delta", 0.0)), float(m.get("max_robot_contact_n", 0.0)) / 150.0],
                        np.float32)
        evs = np.zeros(S.N_EVENTS, np.float32)
        emask = np.zeros(S.N_EVENTS, bool)
        evs[0], emask[0] = float(out.success), True
        evs[2], emask[2] = float(m.get("drops", 0) > before.get("drops", 0)), True
        evs[4], emask[4] = float(m.get("collisions", 0) > before.get("collisions", 0)), True
        evs[5], emask[5] = float(out.result.status in ("guard_tripped", "timeout", "unreachable")), True
        evs[6], emask[6] = float(_num(m.get("damaged", 0)) > _num(before.get("damaged", 0))), True
        for name, j in (("slip", 1), ("jam", 3)):
            if name in m:
                evs[j], emask[j] = float(bool(m[name])), True
        self.lat = map_labels(out.labels, tids)
        inter = S.Interaction(episode_id=getattr(self.env, "episode_id", ""), step=self.k, family=self.task.family,
                              robot=self.task.robot, call_vec=_safe_vec(rcall), skill=call.skill,
                              params=rcall.params, target_is_query=True, segment=seg,
                              effort_cost=out.result.effort.cost(), effort=out.result.effort.as_dict(), t_end=t1,
                              outcome_cont=cont, outcome_cont_mask=np.ones(S.N_CONT, bool), outcome_events=evs,
                              outcome_events_mask=emask, labels={},
                              target_id=tids[0] if len(tids) == 1 else f"pile:{target}")
        out.result.log["interaction"] = inter
        out.result.log["latents"] = dict(self.lat)     # evaluator-only copy (harness/training), never actor input
        self.k += 1
        return out


S_PROP_SIZE = 3  # schema.Prop.SIZE


def map_labels(labels: dict, target_ids) -> dict:
    """ENVS evaluator labels -> spec latent dict for the queried target (pile: total mass, mean friction)."""
    hid = labels.get("hidden", {})
    rows = [hid[t] for t in target_ids if t in hid]
    out = {}
    # fit / articulation latents live on the mating or articulated entity (hole, drawer): merge when unique
    for key in ("clearance", "damping", "frictionloss"):
        vals = [h[key] for h in hid.values() if isinstance(h, dict) and np.isscalar(h.get(key, None))]
        if len(vals) == 1:
            out[HIDDEN_MAP[key]] = float(vals[0])
    th = labels.get("task_hidden", {}) or {}
    for key in ("clearance", "damping", "frictionloss"):      # task-level fit/articulation latents (ENVS task_hidden)
        if np.isscalar(th.get(key, None)) and HIDDEN_MAP[key] not in out:
            out[HIDDEN_MAP[key]] = float(th[key])
    if not rows:
        return out
    if len(rows) == 1:
        for k, v in rows[0].items():
            if k in HIDDEN_MAP and np.isscalar(v) and v is not None and not isinstance(v, str):
                v = float(v)
                if k in ("crush_n", "dent_n") and not np.isfinite(v):
                    v = NOT_FRAGILE_N          # unbreakable part: a large finite limit, so the latent learns the mixture
                out[HIDDEN_MAP[k]] = v
    else:
        if all("mass" in r for r in rows):
            out["mass"] = float(sum(r["mass"] for r in rows))
        mus = [r["mu"] for r in rows if "mu" in r]
        if "mu_mean" in labels:
            out["bulk_friction"] = float(labels["mu_mean"])
        elif mus:
            out["bulk_friction"] = float(np.mean(mus))
    return out


def collect_episode(env, task, policy="mixed", rng=None, max_steps=None):
    """roll out one ENVS episode with a data-collection policy -> (family, robot, steps) for build_family_arrays.
    policy 'mixed': 50% env scripted, 50% random candidates (diverse coverage incl. failures)."""
    ad = EnvAdapter(env)
    ad.reset(task)
    rng = rng or np.random.default_rng(task.seed)
    steps = []
    for k in range(max_steps or getattr(env, "budget", 8)):
        call = None
        if policy in ("mixed", "scripted") and (policy == "scripted" or rng.random() < 0.5):
            call = env.scripted()
        if call is None:
            cs = env.candidate_skills(None, 8, rng)
            if not cs:
                break
            call = cs[int(rng.integers(len(cs)))]
        frame = ad.frame(call.target)
        out = ad.execute(call, relative=False)
        it = out.result.log["interaction"]
        it.labels = dict(out.result.log["latents"])
        steps.append(dict(frame=frame, inter=it))
        if out.done:
            break
    return task.family, task.robot, steps


# ----------------------------------------------------------------- offline conversion of stored ENVS episodes
class _ReplayEnv:
    """minimal env stand-in for EnvAdapter's actor-legal helpers during offline replay (time cursor + held object,
    the latter tracked from the actor's OWN successful picks/places, never from labels)."""

    def __init__(self):
        self.t = 0.0
        self.held = None


_SPEC_CACHE = {}


def robot_spec(name):
    """EmbodimentSpec for a robot name (ROBOTS registry; ENVS test-only probe)."""
    if name in _SPEC_CACHE:
        return _SPEC_CACHE[name]
    spec = None
    try:
        import physmem.embodiments as E
        spec = E.make(name).spec
    except Exception:
        pass
    if spec is None:   # ENVS test-only gantry (docs/ENVS.md)
        spec = api.EmbodimentSpec(name=name, kind="arm", effectors=("hand",), gripper="parallel", reach_m=1.0,
                                  payload_kg=20.0, max_ee_speed=1.0, max_ee_force=250.0, dof=6, floating_base=False,
                                  source="envs-probe")
    _SPEC_CACHE[name] = spec
    return spec


def _safe_vec(call):
    """legacy api.SkillCall.vector() (kept only for compatibility; the model uses spec.named_call_vector)."""
    try:
        return call.vector()
    except (TypeError, ValueError):
        return np.zeros(S.CALL_DIM, np.float32)


def _num(x, default=0.0):
    """robust numeric read of an ENVS log field (some carry reason strings, e.g. metrics['fatal'] = 'lift failed twice')."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def convert_episode(ep: dict):
    """stored ENVS episode dict (physmem.envs.io) -> (family, robot, steps[dict(frame, inter)]) with labels."""
    task = ep["task"]
    fam, robot = task["family"], task["robot"]
    ad = EnvAdapter.__new__(EnvAdapter)
    ad.env = _ReplayEnv()
    ad.task = api.TaskSpec(fam, task.get("category", ""), task["seed"], robot, task["split"], task.get("params", {}))
    ad.idx = EventIndex()
    ad.idx.add(ep["events"])
    ad.k, ad.lat = 0, {}
    ad.pool = []
    steps, prev = [], {}
    for k, st in enumerate(ep["steps"]):
        c = st["call"]
        call = api.SkillCall(c["skill"], dict(c["params"]), target=c.get("target"), effector=c.get("effector"))
        ad.pool = [call]
        ad.env.t = _num(st["t_start"])
        tids = ad.target_ids(call.target)
        frame = ad.frame(call.target)
        p0 = ad.target_pos(call.target)
        seg = build_segment(ad.idx, _num(st["t_start"]), _num(st["t_end"]), tids, p0)
        ad.env.t = _num(st["t_end"])
        p1 = ad.target_pos(call.target)
        m = st["metrics"]
        peak = float(np.abs(seg[:, 12:15]).max()) if len(seg) else 0.0
        cont = np.array([*(p1 - p0), _num(m.get("progress", 0.0)), peak / 50.0, _num(st.get("duration_s", 0.0)),
                         _num(m.get("progress_delta", 0.0)), _num(m.get("max_robot_contact_n", 0.0)) / 150.0],
                        np.float32)
        evs = np.zeros(S.N_EVENTS, np.float32)
        emask = np.ones(S.N_EVENTS, bool)
        evs[0] = float(st.get("success", False) or m.get("success", False))
        evs[1] = float(bool(m.get("slip", False)))
        evs[2] = float(bool(m.get("drop", False)) or _num(m.get("drops", 0)) > _num(prev.get("drops", 0)))
        evs[3] = float(bool(m.get("jam", False)))
        evs[4] = float(bool(m.get("collision", False)) or _num(m.get("collisions", 0)) > _num(prev.get("collisions", 0)))
        evs[5] = float(bool(m.get("stall", False)) or st.get("status") in ("guard_tripped", "timeout", "unreachable"))
        evs[6] = float(_num(m.get("damaged", 0)) > _num(prev.get("damaged", 0)))
        prev = m
        eff = st.get("effort", {})
        rcall = replace(call, params=rel_params(call.params, p0))
        tid = tids[0] if len(tids) == 1 else f"pile:{call.target}"
        inter = S.Interaction(episode_id=ep.get("episode_id", ""), step=k, family=fam, robot=robot,
                              call_vec=_safe_vec(rcall), skill=call.skill, params=rcall.params, target_is_query=True,
                              segment=seg, effort_cost=_num(eff.get("cost", 0.0)), effort=dict(eff),
                              t_end=_num(st["t_end"]), outcome_cont=cont, outcome_cont_mask=np.ones(S.N_CONT, bool),
                              outcome_events=evs, outcome_events_mask=emask,
                              labels=map_labels(st.get("labels", {}), tids), target_id=tid)
        steps.append(dict(frame=frame, inter=inter))
        # actor-side held-object bookkeeping (own commands + own result status)
        if call.skill == "pick" and st.get("ok", False) and not evs[2]:
            ad.env.held = tids[0] if len(tids) == 1 else None
        elif call.skill in ("place", "handover"):
            ad.env.held = None
    return fam, robot, steps
