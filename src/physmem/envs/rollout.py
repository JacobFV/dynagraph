"""Episode runner shared by the generator and tests (agent ENVS).

EpisodeTimeout derives from BaseException ON PURPOSE: skill implementations (e.g. the ROBOTS adapter) catch
`Exception` and turn it into a failed SkillResult, which used to let a wall-clock timeout fire mid-skill and the
episode continue from a broken state (ROBOTS round 5: 3 integrity violations). A BaseException passes through
those handlers; run_episode then ends the episode at once (no further calls) and marks it `aborted`.
"""
from __future__ import annotations

import hashlib
import signal
import time
from dataclasses import asdict, replace

import numpy as np

from . import make_env


class EpisodeTimeout(BaseException):
    pass


JITTER_KEYS = ("speed", "force", "grip", "suction", "lift", "depth", "approach", "release_h", "search", "compliance",
               "stop_time", "distance", "travel", "rate", "timeout", "height")


def perturb(call, rng, frac=0.3):
    """jitter the scripted call's speed / force / grasp / approach params by U(1-frac, 1+frac) (positive scalars)."""
    from dataclasses import replace
    p = dict(call.params)
    for k, v in p.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and any(j in k for j in JITTER_KEYS):
            p[k] = float(v * rng.uniform(1 - frac, 1 + frac))
    return replace(call, params=p)


def choose(env, rng, mode, probe=False):
    """policy mixture: 'scripted' (pure baseline), 'scripted_probe' (r6 fragility-aware probe-then-act scripted
    variant, StationEnv.probe_then_act), 'mixed' (0.4 scripted / 0.4 random feasible candidate / 0.2 perturbed
    scripted; with probe=True the scripted calls are probe-then-act), 'explore' (random candidates).
    Returns (call, policy_source) or (None, None)."""
    if mode == "scripted":
        c = env.scripted()
        return c, "scripted"
    if mode == "scripted_probe":
        c = env.scripted()
        return (env.probe_then_act(c) if c is not None else None), "scripted_probe"
    u = rng.random() if mode == "mixed" else 0.5
    if u >= 0.4 and u < 0.8 or mode == "explore":
        cands = env.candidate_skills(None, 8, rng)
        if cands:
            return cands[int(rng.integers(len(cands)))], "random_candidate"
    c = env.scripted()
    if c is None:
        cands = env.candidate_skills(None, 8, rng)
        return (cands[int(rng.integers(len(cands)))], "random_candidate") if cands else (None, None)
    tag = ""
    if probe:
        if hasattr(env, "force_ramp"):          # r7 tight_insert: the press-until-click variant
            c, tag = env.force_ramp(c), "_ramp"
        else:
            c, tag = env.probe_then_act(c), "_probe"
    if u >= 0.8:
        return perturb(c, rng), "perturbed_scripted" + tag
    return c, "scripted" + tag


def run_episode(task, keep_events=True):
    t0 = time.time()
    env = make_env(task.family)
    ev0 = env.reset(task)
    mode = task.params.get("policy", "scripted")
    prng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"policy|{task.family}|{task.seed}|{task.robot}|{task.split}"
                                                          .encode()).digest()[:8], "little"))
    steps = []
    out = None
    err = None
    aborted = False
    # r6: half of the mixed episodes run the probe-then-act scripted variant (light first squeeze -> tactile
    # stiffness -> crush-bounded grip); drawn from its own stream so the policy draws below are unchanged
    probe = mode == "mixed" and np.random.default_rng(int.from_bytes(hashlib.sha256(
        f"probe|{task.family}|{task.seed}|{task.robot}|{task.split}".encode()).digest()[:8], "little")).random() < 0.5
    try:
        while True:
            call, src = choose(env, prng, mode, probe)
            if call is None:
                break
            t_start = env.t
            out = env.execute(call)
            log = {k: v for k, v in out.result.log.items()}
            steps.append(dict(call=asdict(call), policy_source=src, t_start=t_start, t_end=env.t, ok=out.result.ok,
                              status=out.result.status, effort=out.result.effort.as_dict(),
                              duration_s=out.result.duration_s, done=out.done, success=out.success,
                              metrics=out.metrics, labels=out.labels, log=log))
            if out.done:
                break
    except EpisodeTimeout:      # abort cleanly at once: never issue another call from a broken state
        err = "timeout"
        aborted = True
    except Exception as e:      # keep failures
        err = repr(e)[:500]
    success = bool(env.solved()) if err is None else False
    from physmem.envs.io import events_to_array
    rec = dict(task=asdict(task), episode_id=env.episode_id, policy=mode, probe_then_act=bool(probe),
               integrity=dict(unstable=bool(getattr(env, "unstable", False)),
                              reasons=list(getattr(env, "unstable_reasons", [])),
                              ft_saturated=int(env.metrics_acc.get("ft_saturated", 0)),
                              sim_instabilities=int(env.metrics_acc.get("sim_instabilities", 0))),
               events=events_to_array(env.events_all) if keep_events else None,
               n_events=len(env.events_all), steps=steps, success=success, sim_t=env.t, wall_s=time.time() - t0,
               final_metrics=(out.metrics if out else {}), error=err, goal=env.goal(),
               reset_attempts=getattr(env, "reset_attempts", 1), aborted=aborted)
    return rec


LIMIT_S = 300




def _alarm(signum, frame):
    raise EpisodeTimeout()


def run_with_timeout(task, limit_s=300, keep_events=True):
    """run_episode under a wall-clock limit (SIGALRM; main thread of the worker process only)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit_s))
    try:
        return run_episode(task, keep_events)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
