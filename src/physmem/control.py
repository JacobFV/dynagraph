"""Shared model-predictive push controller (agent G1). Used by ALL models.

    generate_candidates(belief_target_pose, goal, capabilities, rng, n) -> [ActionCandidate]
    select_action(predict_fn, events, candidates, goal, now, ...)       -> TaskAction
    run_episode(world, predictor, controller, seed, ...)                -> EpisodeResult

`predict_fn` is ANY callable (EventTensors, QueryTensors) -> ModelOutput. The
controller only ever hands it tensors built from actor-available
ObservationEvents (never a simulator handle, never labels). Swapping the
predictor changes nothing else in the control path.

Scoring (lower is better), fixed constants logged with every decision:
    score = || goal_xy - world(mean[last horizon, dx:dy]) ||
            + UNCERTAINTY_LAMBDA * sqrt(var_dx + var_dy)    (last horizon)
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from .schema import (
    HORIZONS,
    ActionCandidate,
    CapabilityDescriptor,
    EntityDescriptor,
    EntityType,
    EpisodeResult,
    EventTensors,
    ModelOutput,
    PredictedRollout,
    Prop,
    Provenance,
    QueryTensors,
    TaskAction,
)

UNCERTAINTY_LAMBDA = 0.5
N_CANDIDATES = 48
SCORE_HORIZON_INDEX = -1  # last horizon (2.0 s after push start)
PHI_RANGE = 0.4  # as worlds.sample_push_action
FAST_SPEED = (0.45, 0.80)
SPEED_RANGE = (0.05, 1.0)  # log-uniform like dataset v2; clipped to capability.max_push_speed
DURATION_RANGE = (0.10, 0.50)  # dataset v2 (data.WORLD_TRAIN_V2 dur)


def _log_uniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
STOP_TOL = 0.015  # controller's own stop rule on its BELIEF (evaluator uses 3 cm on truth)


# --------------------------------------------------------------------------
# tensors (agent DATA's tensorizer when available, else a minimal private adapter)
# --------------------------------------------------------------------------


def _tensorize(events, now, target_id, rng, B):
    """Build EventTensors for one event history replicated B times + the canonical
    frame (x0, y0, yaw0) = latest available pose estimate of the target."""
    import torch

    try:
        from . import data as D

        et, info = D.events_to_tensors([events], [now], rng, permute=True, targets=[target_id], return_info=True)
        tidx = int(info["target_index"][0])
        est = tuple(float(v) for v in info["est_pose"][0])
        make_query = D.make_query
        adapter = "data.events_to_tensors"
    except ImportError:  # pragma: no cover - fallback before data.py existed
        et, tidx, est = _minimal_tensorize(events, now, target_id)
        make_query = None
        adapter = "control._minimal_tensorize"
    rep = {}
    for k, v in et.__dict__.items():
        if hasattr(v, "shape") and v.dim() >= 1 and v.shape[0] == 1:
            rep[k] = v.expand(B, *v.shape[1:]).contiguous()
        else:
            rep[k] = v
    if et.dt_obs.dim() == 1 and et.dt_obs.shape[0] == 1:
        rep["dt_obs"] = et.dt_obs.expand(B).contiguous()
    return EventTensors(**rep), tidx, est, make_query, adapter


def _minimal_tensorize(events, now, target_id):  # pragma: no cover - superseded by data.py
    raise ImportError("data.py unavailable and minimal adapter disabled")


def _query(make_query, tidx, candidates):
    import torch

    a = np.stack([c.vector() for c in candidates]).astype(np.float32)
    B = len(candidates)
    if make_query is not None:
        return make_query(np.full(B, tidx), a)
    return QueryTensors(target=torch.full((B,), tidx, dtype=torch.long), action=torch.from_numpy(a),
                        horizons=torch.tensor(np.tile(np.asarray(HORIZONS, np.float32), (B, 1))))


# --------------------------------------------------------------------------
# stub predictor (prior-friction analytic Coulomb push model)
# --------------------------------------------------------------------------


def _push_geometry(half, theta, phi):
    ux, uy = math.cos(theta), math.sin(theta)
    hx, hy = half[0], half[1]
    sx = hx / abs(ux) if abs(ux) > 1e-9 else np.inf
    sy = hy / abs(uy) if abs(uy) > 1e-9 else np.inf
    if sx < sy:
        p, n = (sx * ux, sx * uy), (math.copysign(1.0, ux), 0.0)
    else:
        p, n = (sy * ux, sy * uy), (0.0, math.copysign(1.0, uy))
    cp, sp = math.cos(phi), math.sin(phi)
    d = (-(cp * n[0] - sp * n[1]), -(sp * n[0] + cp * n[1]))
    return np.array(p), np.array(d), np.array(n)


class PriorCoulombStub:
    """Stub predictor: object moves with the pusher along the push direction after
    the approach gap closes, then Coulomb sliding with a FIXED prior friction.
    Reads only the target SIZE from the event tensors (actor evidence). Variance:
    constant + grows with predicted slide distance (friction uncertainty)."""

    def __init__(self, mu: float = 0.5, gap: float = 0.025, name: Optional[str] = None):
        self.mu = mu
        self.gap = gap
        self.name = name or f"stub_prior_coulomb_mu{mu:g}"

    def __call__(self, events: EventTensors, query: QueryTensors, **_) -> ModelOutput:
        import torch

        a = query.action.detach().cpu().numpy()
        hz = query.horizons.detach().cpu().numpy()
        tgt = query.target.detach().cpu().numpy()
        ent = events.entity.cpu().numpy()
        prop = events.prop.cpu().numpy()
        val = events.value.cpu().numpy()
        mask = events.mask.cpu().numpy()
        B, H = hz.shape
        mean = np.zeros((B, H, 7), np.float32)
        logvar = np.zeros((B, H, 7), np.float32)
        g = 9.81
        for b in range(B):
            m = mask[b] & (ent[b] == tgt[b]) & (prop[b] == int(Prop.SIZE))
            half = np.abs(val[b][np.nonzero(m)[0][-1], :3]) if m.any() else np.array([0.04, 0.04, 0.02])
            th, ph = math.atan2(a[b, 1], a[b, 0]), math.atan2(a[b, 3], a[b, 2])
            sp, du = float(a[b, 4]), float(a[b, 5])
            _, d, n = _push_geometry(half, th, ph)
            cos_in = max(abs(float(d @ -n)), 0.2)
            tc = min(self.gap / cos_in / max(sp, 1e-3), du)
            for i, h in enumerate(hz[b]):
                if h <= tc:
                    s, v = 0.0, 0.0
                elif h <= du:
                    s, v = sp * (h - tc), sp
                else:
                    v0 = sp if du > tc else 0.0
                    ts = v0 / (self.mu * g)
                    tt = min(h - du, ts)
                    s = sp * (du - tc) + v0 * tt - 0.5 * self.mu * g * tt * tt
                    v = max(0.0, v0 - self.mu * g * (h - du))
                mean[b, i, 0:2] = d * s
                mean[b, i, 2] = 1.0
                mean[b, i, 4:6] = d * v
                slide = sp * sp / (2 * self.mu * g)
                logvar[b, i, :] = math.log((0.01 + 0.5 * slide) ** 2)
        return ModelOutput(mean=torch.from_numpy(mean), logvar=torch.from_numpy(logvar),
                           friction_mean=torch.full((B,), float(self.mu)), friction_logvar=torch.full((B,), math.log(0.04)))


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


def _start_end_world(pose, half, params, r_eff, gap):
    x, y, c, s = pose[:4]
    yaw = math.atan2(s, c)
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    th, ph, sp, du = params
    p, d, n = _push_geometry(half, th, ph)
    st = np.array([x, y]) + R @ (p + n * (r_eff + gap))
    en = st + (R @ d) * sp * du
    return st, en, R @ d


def generate_candidates(belief_target_pose, goal, capabilities: CapabilityDescriptor, rng: np.random.Generator,
                        n: int = N_CANDIDATES, half=(0.04, 0.04, 0.02), target_entity_id: str = "obj",
                        gap: float = 0.025, goal_directed_frac: float = 0.6, fast_frac: float = 0.0,
                        fast_speed=FAST_SPEED, max_stroke: Optional[float] = None) -> list[ActionCandidate]:
    """Sample push candidates in the OBJECT frame of the belief pose. A fraction are
    aimed so the push direction points at the goal (with jitter); the rest are
    uniform over the boundary. Feasibility: start and end points of the effector
    inside the capability workspace, speed/duration within limits."""
    x, y, c, s = belief_target_pose[:4]
    yaw = math.atan2(s, c)
    gx, gy = goal["target_xy"]
    gdir_w = math.atan2(gy - y, gx - x)
    gdir_l = gdir_w - yaw
    dist = math.hypot(gx - x, gy - y)
    out = []
    n_fast = int(round(fast_frac * n))
    vmax = min(SPEED_RANGE[1], capabilities.max_push_speed)
    for i in range(n):
        if i < n_fast and fast_speed[0] <= vmax:
            # goal-directed FAST pushes (contact + free slide), so long slides are always on the menu
            th = gdir_l + math.pi + rng.normal(0, 0.3)
            ph = float(np.clip(rng.normal(0, 0.12), -PHI_RANGE, PHI_RANGE))
            sp = float(rng.uniform(fast_speed[0], min(fast_speed[1], vmax)))
            du = float(np.clip(rng.uniform(0.15, 1.0) * dist / sp, *DURATION_RANGE))
        elif i < n_fast + int(goal_directed_frac * (n - n_fast)):
            # contact on the side opposite the goal direction, phi jitter
            th = gdir_l + math.pi + rng.normal(0, 0.35)
            ph = float(np.clip(rng.normal(0, 0.15), -PHI_RANGE, PHI_RANGE))
            # durations/speeds spanning short taps to long pushes relative to the distance
            sp = _log_uniform(rng, SPEED_RANGE[0], min(SPEED_RANGE[1], capabilities.max_push_speed))
            du = float(np.clip(rng.uniform(0.2, 1.4) * (dist + gap) / sp, *DURATION_RANGE))
        else:
            th = float(rng.uniform(-math.pi, math.pi))
            ph = float(rng.uniform(-PHI_RANGE, PHI_RANGE))
            sp = _log_uniform(rng, SPEED_RANGE[0], min(SPEED_RANGE[1], capabilities.max_push_speed))
            du = float(rng.uniform(*DURATION_RANGE))
        th = (th + math.pi) % (2 * math.pi) - math.pi
        sp = min(sp, capabilities.max_push_speed)
        du = min(du, capabilities.max_push_duration)
        if max_stroke is not None:  # hand travel per push limited: v * T <= max_stroke
            du = float(np.clip(min(du, max_stroke / sp), DURATION_RANGE[0], DURATION_RANGE[1]))
            sp = min(sp, max_stroke / du)
        params = (float(th), float(ph), float(sp), float(du))
        st, en, _ = _start_end_world(belief_target_pose, half, params, capabilities.effector_radius, gap)
        lo, hi = np.array(capabilities.workspace_min), np.array(capabilities.workspace_max)
        ok = bool(np.all(st >= lo) and np.all(st <= hi) and np.all(en >= lo) and np.all(en <= hi))
        out.append(ActionCandidate(candidate_id=f"c{i:02d}", kind="push", params=params, target_entity_id=target_entity_id,
                                   feasible=ok, feasibility_note="" if ok else "effector path outside workspace"))
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@dataclass
class ControllerConfig:
    n_candidates: int = N_CANDIDATES
    uncertainty_lambda: float = UNCERTAINTY_LAMBDA
    max_pushes: int = 6
    cap_s: float = 20.0
    stop_tol: float = STOP_TOL
    settle_speed: float = 0.01
    settle_window: float = 0.25
    max_settle_wait: float = 1.5
    final_hold: float = 0.7
    ik_retries: int = 64
    # objective (all constants logged with every episode)
    sd_mode: str = "xy"  # "xy": sqrt(var_x+var_y) of final position; "dist": sd of predicted distance-to-goal
    push_cost: float = 0.0  # m-equivalent penalty per (anticipated) extra push
    success_tol: float = 0.03
    fast_frac: float = 0.0  # fraction of candidates drawn from FAST_SPEED
    max_stroke: Optional[float] = None  # max hand travel per push (m)
    benchmark: str = "tabletop-push-easy"


EASY = ControllerConfig()
SLIDE_TO_TARGET = ControllerConfig(max_pushes=3, cap_s=15.0, sd_mode="dist", push_cost=0.05, fast_frac=0.4,
                                   max_stroke=0.08, benchmark="slide-to-target")


def score_rollouts(out: ModelOutput, est, goal, lam, sd_mode="xy", push_cost=0.0, tol=0.03):
    """score = predicted final distance to goal + lam * sd + push_cost * 1[predicted dist > tol].
    sd_mode 'xy': sqrt(var_x + var_y) of the final position; 'dist': sd of the predicted DISTANCE to
    the goal (variance projected on the goal direction), so spread that does not change the distance
    (e.g. lateral) is not penalised. Variances are the model's own (no per-model recalibration)."""
    mean = out.mean.detach().float().cpu().numpy().astype(np.float64)
    logvar = out.logvar.detach().float().cpu().numpy().astype(np.float64)
    x0, y0, yaw0 = est
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = mean[:, SCORE_HORIZON_INDEX, 0], mean[:, SCORE_HORIZON_INDEX, 1]
    wx, wy = x0 + c * dx - s * dy, y0 + s * dx + c * dy
    gx, gy = goal["target_xy"]
    dist = np.hypot(gx - wx, gy - wy)
    vx = np.exp(logvar[:, SCORE_HORIZON_INDEX, 0])
    vy = np.exp(logvar[:, SCORE_HORIZON_INDEX, 1])
    if sd_mode == "dist":
        ux, uy = (gx - wx) / np.maximum(dist, 1e-6), (gy - wy) / np.maximum(dist, 1e-6)
        lx, ly = c * ux + s * uy, -s * ux + c * uy  # goal direction in the canonical frame
        sd = np.sqrt(lx * lx * vx + ly * ly * vy)
    else:
        sd = np.sqrt(vx + vy)
    score = dist + lam * sd + push_cost * (dist > tol)
    return score, dist, sd, np.stack([wx, wy], 1), mean, logvar


def select_action(predict_fn: Callable, events, candidates, goal, now: float, target_id: str, rng=None,
                  lam: float = UNCERTAINTY_LAMBDA, belief_revision: int = -1, exclude=(), sd_mode: str = "xy",
                  push_cost: float = 0.0, tol: float = 0.03) -> tuple[TaskAction, dict]:
    """Batch-predict every feasible candidate in ONE call and pick the min score.
    Returns (TaskAction, info) where info has per-candidate scores and latency."""
    rng = rng or np.random.default_rng(0)
    feas = [c for c in candidates if c.feasible and c.candidate_id not in exclude]
    if not feas:
        raise RuntimeError("no feasible candidates")
    et, tidx, est, mq, adapter = _tensorize(events, now, target_id, rng, len(feas))
    q = _query(mq, tidx, feas)
    t0 = time.perf_counter()
    out = predict_fn(et, q)
    latency = time.perf_counter() - t0
    score, dist, sd, wxy, mean, logvar = score_rollouts(out, est, goal, lam, sd_mode, push_cost, tol)
    j = int(np.argmin(score))
    ch = feas[j]
    roll = PredictedRollout(candidate_id=ch.candidate_id, belief_revision=belief_revision, belief_time=now,
                            horizons=tuple(HORIZONS), mean=mean[j], var=np.exp(logvar[j]),
                            aux=dict(canonical_frame=est))
    if getattr(out, "friction_mean", None) is not None:
        fm = out.friction_mean.detach().cpu().numpy()
        fl = out.friction_logvar.detach().cpu().numpy() if out.friction_logvar is not None else np.zeros_like(fm)
        roll.aux["friction"] = (float(fm[j]), float(np.exp(fl[j])))
    all_scores = {c.candidate_id: float(sc) for c, sc in zip(feas, score)}
    order = [feas[i].candidate_id for i in np.argsort(score)]
    info = dict(latency_s=latency, adapter=adapter, est=est, order=order, n_feasible=len(feas),
                dist=dict(zip([c.candidate_id for c in feas], dist.tolist())),
                sd=dict(zip([c.candidate_id for c in feas], sd.tolist())),
                pred_world_xy={c.candidate_id: wxy[i].tolist() for i, c in enumerate(feas)},
                rollouts={c.candidate_id: mean[i] for i, c in enumerate(feas)},
                rollout_var={c.candidate_id: np.exp(logvar[i]) for i, c in enumerate(feas)})
    return TaskAction(candidate=ch, score=float(score[j]), rollout=roll, all_scores=all_scores), info


# --------------------------------------------------------------------------
# closed loop
# --------------------------------------------------------------------------


def _to_canonical(xy, yaw, est):
    x0, y0, yaw0 = est
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = xy[0] - x0, xy[1] - y0
    return np.array([c * dx + s * dy, -s * dx + c * dy, math.cos(yaw - yaw0), math.sin(yaw - yaw0)])


def run_episode(world, predictor: Callable, controller: Optional[ControllerConfig] = None, seed: int = 0,
                model_name: Optional[str] = None, on_decision: Optional[Callable] = None,
                on_select: Optional[Callable] = None) -> EpisodeResult:
    """Closed loop: observe -> belief -> predict all candidates (one batch) -> select ->
    execute the push with the robot -> repeat until the controller's belief says the
    object is at the goal and still, or max pushes / sim-time cap.

    `world` must expose: actor (robot: drain_events, plan_push, execute_push, hold, port.time),
    capability, goal, target_entity, ep (with seed), and EVALUATOR-ONLY methods
    push_success(), true_object(), trace(), audit(). The predictor only sees tensors."""
    from .memory import WorldBelief

    cfg = controller or ControllerConfig()
    rng = np.random.default_rng(seed)
    robot = world.actor
    port = robot.port
    model_name = model_name or getattr(predictor, "name", type(predictor).__name__)
    belief = WorldBelief(port.episode_id, actor_boundary=True)
    belief.add_entity(EntityDescriptor(world.target_entity, EntityType.OBJECT))
    tgt = world.target_entity
    history: list = []
    decisions = []
    wall0 = time.perf_counter()
    infer_total = 0.0

    def observe():
        evs = robot.drain_events()
        for e in evs:
            if e.provenance == Provenance.PRIVILEGED_LABEL:
                raise ValueError("privileged event at actor boundary")
        history.extend(evs)
        belief.ingest_events(evs, now=port.time)

    def belief_pose():
        m, v, rec = belief.resolve_query(tgt, Prop.POSE2)
        return m, v

    def belief_speed():
        try:
            m, _, _ = belief.resolve_query(tgt, Prop.VEL2)
            return float(math.hypot(m[0], m[1]))
        except KeyError:
            return 0.0

    def wait_settled():
        t_start = port.time
        still_since = None
        while port.time - t_start < cfg.max_settle_wait:
            robot.hold(0.05, "settle")
            observe()
            # smoothed speed from the last 0.25 s of pose evidence (VEL2 is noisy)
            sp = _recent_speed(history, tgt, port.time, cfg.settle_window)
            if sp < cfg.settle_speed:
                still_since = still_since if still_since is not None else port.time
                if port.time - still_since >= cfg.settle_window - 1e-9:
                    return
            else:
                still_since = None

    seen_rev: dict = {}

    def memory_updates():
        """Belief records changed since the previous decision (WorldBelief.snapshot)."""
        snap = belief.snapshot()
        out = []
        for r in snap["records"]:
            key = (r["entity"], r["prop"])
            if seen_rev.get(key) != r["rev"]:
                seen_rev[key] = r["rev"]
                out.append(dict(entity=r["entity"], prop=r["prop"], mean=r["mean"], var=r["var"], t=r["t"],
                                revision=r["rev"], kind=r["kind"], sources=r["sources"], refs=r["refs"]))
        return dict(belief_revision=snap["revision"], records=out)

    robot.hold(0.3, "observe")
    observe()
    n_push = 0
    status = "running"
    while True:
        wait_settled()
        pose, pvar = belief_pose()
        gx, gy = world.goal["target_xy"]
        bdist = math.hypot(pose[0] - gx, pose[1] - gy)
        if bdist <= cfg.stop_tol:
            status = "belief_at_goal"
            break
        if n_push >= cfg.max_pushes:
            status = "max_pushes"
            break
        if port.time >= cfg.cap_s - 2.0:
            status = "time_cap"
            break
        size, _, _ = belief.resolve_query(tgt, Prop.SIZE)
        half = tuple(float(v) for v in size[:3])
        cands = generate_candidates(pose, world.goal, world.capability, rng, n=cfg.n_candidates, half=half,
                                    target_entity_id=tgt, fast_frac=cfg.fast_frac, max_stroke=cfg.max_stroke)
        now = port.time
        events_now = belief.available_events(now)
        w_inf = time.perf_counter()
        if not any(c.feasible for c in cands):
            status = "no_feasible_candidate"
            break
        action, info = select_action(predictor, events_now, cands, world.goal, now, tgt, rng=rng,
                                     lam=cfg.uncertainty_lambda, belief_revision=belief.revision,
                                     sd_mode=cfg.sd_mode, push_cost=cfg.push_cost, tol=cfg.success_tol)
        infer_total += time.perf_counter() - w_inf
        # robot-specific reachability (IK on the robot-only model); fall back down the ranking
        chosen, plan, rejected = None, None, []
        for cid in info["order"][: cfg.ik_retries]:
            c = next(c for c in cands if c.candidate_id == cid)
            p = robot.plan_push(c, pose, half)
            if p["feasible"]:
                chosen, plan = c, p
                break
            rejected.append(cid)
        if chosen is None:
            status = "no_reachable_candidate"
            break
        est = info["est"]
        mem_upd = memory_updates()
        st_before = world.true_object()  # evaluator-only (logged, never given to actor)
        pm = info["rollouts"][chosen.candidate_id]
        pv = info["rollout_var"][chosen.candidate_id]
        c0, s0 = math.cos(est[2]), math.sin(est[2])
        pred_world = [[est[0] + c0 * r[0] - s0 * r[1], est[1] + s0 * r[0] + c0 * r[1]] for r in pm]
        pred_sd = [float(math.sqrt(v[0] + v[1])) for v in pv]
        if on_select is not None:
            on_select(dict(k=n_push, chosen=chosen.candidate_id, params=list(chosen.params), pred_world=pred_world,
                           pred_sd=pred_sd, belief_pose=[float(v) for v in pose], belief_var=[float(v) for v in pvar],
                           friction=action.rollout.aux.get("friction"), model=model_name, est=list(est),
                           score=action.all_scores[chosen.candidate_id], half=half, memory_updates=mem_upd,
                           belief_time=now, belief_revision=belief.revision))
        exec_info = robot.execute_push(chosen, pose, half, plan)
        n_push += 1
        observe()
        dec = dict(
            k=n_push - 1, memory_updates=mem_upd, belief_time=now, belief_revision=belief.revision, belief_pose=[float(v) for v in pose],
            belief_pose_var=[float(v) for v in pvar], canonical_frame=list(est), goal=list(world.goal["target_xy"]),
            candidates=[dict(id=c.candidate_id, params=list(c.params), feasible=c.feasible, note=c.feasibility_note,
                             score=action.all_scores.get(c.candidate_id), pred_dist=info["dist"].get(c.candidate_id),
                             pred_sd=info["sd"].get(c.candidate_id),
                             pred_world_xy=info["pred_world_xy"].get(c.candidate_id)) for c in cands],
            chosen=chosen.candidate_id, chosen_params=list(chosen.params), model_top=action.candidate.candidate_id,
            ik_rejected=rejected, score=action.all_scores[chosen.candidate_id],
            predicted_world_xy=pred_world, predicted_sd_xy=pred_sd,
            predicted_mean=info["rollouts"][chosen.candidate_id].tolist(),
            predicted_var=info["rollout_var"][chosen.candidate_id].tolist(), horizons=list(HORIZONS),
            friction_estimate=action.rollout.aux.get("friction") if chosen.candidate_id == action.candidate.candidate_id else None,
            inference_latency_s=info["latency_s"], tensor_adapter=info["adapter"], lambda_=cfg.uncertainty_lambda,
            execution=exec_info, plan=dict(side=plan["side"], lead=plan["lead"], ik_err=plan["ik_err"]),
            truth_before_eval_only=dict(x=float(st_before["pos"][0]), y=float(st_before["pos"][1]), yaw=st_before["yaw"]),
        )
        decisions.append(dec)
        if on_decision is not None:
            on_decision(dec)
    robot.hold(cfg.final_hold, "final")
    observe()
    # ---------------- evaluator side (privileged) ----------------
    succ = world.push_success()
    trace = world.trace()
    for dec in decisions:  # realized outcome at the horizons, canonical frame of the decision
        tp = dec["execution"]["t_push"]
        nxt = [d2["execution"]["t_start"] for d2 in decisions if d2["k"] == dec["k"] + 1]
        t_lim = nxt[0] if nxt else float("inf")
        real = []
        for h in HORIZONS:
            t = tp + h
            if t > t_lim or t > trace[-1]["t"]:
                real.append(None)
                continue
            r = min(trace, key=lambda z: abs(z["t"] - t))
            real.append(_to_canonical((r["x"], r["y"]), r["yaw"], dec["canonical_frame"]).tolist())
        dec["realized_canonical_eval_only"] = real
        pm = np.array(dec["predicted_mean"])
        errs = [float(np.hypot(pm[i, 0] - rr[0], pm[i, 1] - rr[1])) for i, rr in enumerate(real) if rr is not None]
        dec["pred_pos_err_at_horizons"] = errs
        t_after = t_lim if nxt else trace[-1]["t"]
        r = min(trace, key=lambda z: abs(z["t"] - t_after))
        gx, gy = world.goal["target_xy"]
        dec["eval_dist_to_goal_after"] = float(math.hypot(r["x"] - gx, r["y"] - gy))
        # predicted final position error vs realized final (settled) position, world frame
        dec["eval_final_pred_err"] = float(math.hypot(dec["predicted_world_xy"][-1][0] - r["x"],
                                                      dec["predicted_world_xy"][-1][1] - r["y"]))
    wall = time.perf_counter() - wall0
    sim_t = port.time
    audit = world.audit()
    metrics = dict(
        status=status, n_pushes=n_push, final_dist=succ["dist"], still=succ["still"], clean=succ["clean"],
        sim_time=sim_t, wall_time=wall, real_time_factor=sim_t / max(wall, 1e-9),
        inference_wall_total=infer_total, mean_inference_latency=float(np.mean([d["inference_latency_s"] for d in decisions])) if decisions else 0.0,
        n_contact_events=sum(1 for e in history if e.property_type == Prop.CONTACT and e.value[0] > 0.5),
        audit=audit, lambda_=cfg.uncertainty_lambda, controller=asdict(cfg),
        dist_after_first_push=decisions[0]["eval_dist_to_goal_after"] if decisions else None,
        benchmark=cfg.benchmark, objective=succ["dist"] + cfg.push_cost * n_push, push_cost=cfg.push_cost,
        chosen_speeds=[d["chosen_params"][2] for d in decisions],
        friction_estimates=[d["friction_estimate"] for d in decisions],
        mean_final_pred_err=float(np.mean([d["eval_final_pred_err"] for d in decisions])) if decisions else None,
        hidden_eval_only=dict(friction=world.ep.box.friction, mass=world.ep.box.mass),
    )
    return EpisodeResult(episode_id=port.episode_id, model=model_name, seed=seed, success=succ["success"],
                         metrics=metrics, decisions=decisions)


def _recent_speed(history, tgt, now, window):
    pts = [(e.measured_at, e.value[0], e.value[1]) for e in history
           if e.entity_id == tgt and e.property_type == Prop.POSE2 and e.measured_at >= now - window - 1e-9]
    if len(pts) < 3:
        return float("inf")
    t = np.array([p[0] for p in pts])
    xy = np.array([[p[1], p[2]] for p in pts])
    A = np.stack([t - t.mean(), np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return float(np.hypot(coef[0, 0], coef[0, 1]))


# --------------------------------------------------------------------------
# predictor registry (all share the same (EventTensors, QueryTensors) -> ModelOutput call)
# --------------------------------------------------------------------------


class _NoGrad:
    def __init__(self, model, name, passes=4, device="cpu"):
        self.model, self.name, self.passes, self.device = model, name, passes, device

    def __call__(self, events, query):
        import torch

        with torch.no_grad():
            return self.model(events.to(self.device), query.to(self.device), passes=self.passes)


def make_predictor(spec: str, oracle_friction: Optional[float] = None, device: str = "cpu"):
    """spec: 'stub[:mu]' | 'analytic' (models.analytic.CoulombAnalytic) | 'cv' (ConstantVelocity) |
    'compact:<ckpt path>[:passes]' | 'oracle' (stub with the TRUE friction -- evaluator-provided,
    labelled ORACLE, upper bound only)."""
    if spec.startswith("stub"):
        mu = float(spec.split(":")[1]) if ":" in spec else 0.5
        return PriorCoulombStub(mu)
    if spec == "oracle":
        if oracle_friction is None:
            raise ValueError("oracle needs the evaluator-provided friction")
        return PriorCoulombStub(oracle_friction, name="ORACLE_true_friction_coulomb")
    if spec in ("analytic", "cv"):
        from .models import analytic as A

        m = A.CoulombAnalytic() if spec == "analytic" else A.ConstantVelocity()
        f = lambda ev, q: m(ev, q)  # noqa: E731
        f.name = "analytic_coulomb_ls_friction" if spec == "analytic" else "constant_velocity"
        return f
    if spec.startswith("inkling:"):
        _, variant, seed = spec.split(":")[:3]
        return InklingPredictor(variant, seed)
    if spec.startswith("compact:"):
        parts = spec.split(":")
        from .evaluate import load_compact
        from pathlib import Path as _P

        m = load_compact(parts[1], device)
        passes = int(parts[2]) if len(parts) > 2 else 4
        return _NoGrad(m, f"compact_{_P(parts[1]).stem}_p{passes}", passes, device)
    raise ValueError(spec)


# --------------------------------------------------------------------------
# Inkling core in the loop (GPU heavy: run under scripts/gpu_heavy.sh)
# --------------------------------------------------------------------------

INKLING_CKPT_DIR = "ckpt/Inkling-2-Mini"
INKLING_CHUNK = 16  # candidates per forward (bs 32 is slow)
_INKLING_TEXT = None  # (tm, tok) loaded once per process
_INKLING_LAST = None  # (spec_key, model): building another variant re-injects LoRA into the shared base


@__import__("functools").lru_cache(maxsize=64)
def _ckpt_val_nll(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    v = ck.get("val")
    if isinstance(v, dict):
        return float(v.get("nll", v.get("total", float("inf"))))
    return float(v) if v is not None else float("inf")


def best_inkling_seed(variant, ckpt_dir="results/checkpoints", tag="main"):
    """Best seed by the checkpoint's validation nll (key 'val'); un-suffixed files only."""
    import glob
    import re

    best = None
    for pth in glob.glob(f"{ckpt_dir}/inkling-{tag}-{variant}-s*.pt"):
        m = re.search(r"-s(\d+)\.pt$", pth)
        if not m:
            continue
        nll = _ckpt_val_nll(pth)
        if best is None or nll < best[0]:
            best = (nll, int(m.group(1)), pth)
    return best


class InklingPredictor:
    """predict_fn wrapper around physmem.models.inkling (i00 / i11). Loaded once per process;
    chunked bf16 inference on CUDA; returns a CPU float32 ModelOutput. The simulator is paused while
    this runs (synchronous); per-decision wall latency is logged by run_episode."""

    label = "inkling core in the loop, sim paused during inference"

    def __init__(self, variant: str, seed, ckpt_dir: str = "results/checkpoints", tag: str = "main",
                 text_path: str = INKLING_CKPT_DIR):
        global _INKLING_TEXT, _INKLING_LAST
        import torch
        from .models import inkling as IK

        if seed == "best":
            b = best_inkling_seed(variant, ckpt_dir, tag)
            if b is None:
                raise FileNotFoundError(f"no inkling-{tag}-{variant} checkpoints")
            seed = b[1]
        seed = int(seed)
        self.variant, self.seed = variant, seed
        self.ckpt = f"{ckpt_dir}/inkling-{tag}-{variant}-s{seed}.pt"
        self.val_nll = _ckpt_val_nll(self.ckpt)
        self.name = f"inkling_{variant}_s{seed}"
        key = (variant, seed, self.ckpt)
        if _INKLING_LAST is not None and _INKLING_LAST[0] == key:
            self.model = _INKLING_LAST[1]
            return
        if _INKLING_TEXT is None:
            tm, tok, _, rep = IK.load_inkling_text(text_path)
            _INKLING_TEXT = (tm, tok)
        tm, tok = _INKLING_TEXT
        prefix = IK.prefix_ids_from_tokenizer(tok).cuda()
        m = IK.build_inkling_model(variant, tm, prefix, seed=seed)
        IK.load_trainable(m, self.ckpt)
        m.eval()
        self.model = m
        _INKLING_LAST = (key, m)

    def __call__(self, events: EventTensors, query: QueryTensors) -> ModelOutput:
        import torch

        B = query.action.shape[0]
        outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for a in range(0, B, INKLING_CHUNK):
                sl = slice(a, min(B, a + INKLING_CHUNK))
                ev = EventTensors(**{k: (v[sl] if hasattr(v, "shape") and v.dim() >= 1 and v.shape[0] == B else v)
                                     for k, v in events.__dict__.items()}).to("cuda")
                q = QueryTensors(**{k: v[sl] for k, v in query.__dict__.items()}).to("cuda")
                outs.append(self.model(ev, q))
        torch.cuda.synchronize()

        def cat(field):
            xs = [getattr(o, field) for o in outs]
            if any(x is None for x in xs):
                return None
            return torch.cat([x.float().reshape(x.shape[0], *x.shape[1:]) for x in xs]).cpu()

        return ModelOutput(mean=cat("mean"), logvar=cat("logvar"), friction_mean=cat("friction_mean"),
                           friction_logvar=cat("friction_logvar"), mass_mean=cat("mass_mean"),
                           mass_logvar=cat("mass_logvar"), contact_logits=cat("contact_logits"))
