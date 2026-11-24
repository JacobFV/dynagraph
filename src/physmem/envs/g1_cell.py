"""envs-r6: G1 (Unitree G1 + Dex3) two-palm tote TRANSFER for the tote_transfer humanoid cell.

ROBOTS' `humanoid.g1_tote_carry` is a lift-hold-put-back primitive: it lifts a tote between both palms, optionally
carries it sideways (`carry_dy`), then carries it BACK and puts it down where it was. A transfer needs the put-down
at the destination, so this module re-uses the same controller calls (palm frames, hybrid force squeeze along the
palm normals, blended trajectories, fall check) with a one-way carry:

    approach (up, wide, beside) -> force-squeeze both palms (squeeze_n each) -> lift -> carry dy -> lower -> release

It runs as the robot's `pick` when the call carries `two_palm=1` (one call moves the tote; env.held stays None).
The only change in ROBOTS' adapter (`envs/robots.py::RobotSkills.sk_pick`) is the 3-line routing to this function;
see docs/ENVS.md "G1 tote cell". The palm frames assume the humanoid faces +x (the cell mounts it at yaw 0), as in
ROBOTS' tote test (`scripts/robots_g1_tasks.py::tote_test`). Actor-legal: only the tote ESTIMATE is used.
"""
from __future__ import annotations

import numpy as np

from ..v2.api import EffortMeter, SkillResult


def _transfer(ctx, est_pos, half, squeeze_n=30.0, lift=0.10, carry_dy=0.0, carry_T=2.0):
    from physmem.embodiments.humanoid import _axes_quat
    from physmem.embodiments.trajectory import BlendedTrajectory
    import mujoco
    emb, c, d = ctx.emb, ctx.ctrl, ctx.data
    est = np.asarray(est_pos, float)
    tote_z = est[2]
    g_saved = dict(c.g)
    c.g['root_z_w'] = 8000.0
    q, ref = {}, {}
    for s, sg in (('left', 1.0), ('right', -1.0)):
        q[s] = _axes_quat((0, -sg, 0), (1.0, 0, 0))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[s])
        ref[s] = np.array([est[0], est[1] + sg * (half[1] + 0.01), tote_z])
    p0 = {s: emb.tcp_pose(d, s)[0] for s in ('left', 'right')}
    c.set_mode('track')
    for s, sg in (('left', 1.0), ('right', -1.0)):
        c.set_gripper(s, 1.0)
        top = tote_z + half[2] + 0.06
        up = np.array([p0[s][0], p0[s][1], top])
        wide = np.array([ref[s][0], ref[s][1] + sg * 0.06, top])
        c.command(BlendedTrajectory(effector=s, waypoints=[(up, q[s]), (wide, q[s]), (ref[s] + np.array([0, sg * 0.06, 0]), q[s])],
                                    durations=[1.2, 1.2, 0.8]))
    ctx.run(max_t=7.0)
    for s, sg in (('left', 1.0), ('right', -1.0)):
        c.set_mode('force', s, axis=(0, -sg, 0), force=float(squeeze_n), approach_speed=0.05, touch_n=2.0)
        c.command(BlendedTrajectory(effector=s, waypoints=[(ref[s], q[s])], durations=[0.8]))
    ctx.run(max_t=2.5)
    ctx.wait(0.8)
    info = dict(touched=bool(all(c.st[s]['touched'] for s in ('left', 'right'))))

    def move_both(dz, dy, T):
        for s in ('left', 'right'):
            pr = c.st[s]['ref'][0] if c.st[s]['ref'] is not None else emb.tcp_pose(d, s)[0]
            c.command(BlendedTrajectory(effector=s, waypoints=[(pr + np.array([0, dy, dz]), q[s])], durations=[T]),
                      preempt=True)
        t_end = d.time + T + 0.3
        while d.time < t_end and not c.fallen(d):
            ctx.step()

    if info['touched']:
        move_both(lift, 0.0, 1.5)
        if carry_dy:
            move_both(0.0, carry_dy, carry_T)
        move_both(-lift + 0.005, 0.0, 1.5)
    for s, sg in (('left', 1.0), ('right', -1.0)):
        c.set_mode('track', s)
        pr = emb.tcp_pose(d, s)[0]
        c.command(BlendedTrajectory(effector=s, waypoints=[(pr + np.array([0, sg * 0.08, 0.04]), q[s])], durations=[1.0]),
                  preempt=True)
    ctx.run(max_t=2.0)
    c.g.clear()
    c.g.update(g_saved)
    info['fell'] = bool(c.fallen(d))
    return info


def two_palm_transfer(env, call, ctx, est):
    """`pick` with params two_palm=1: squeeze_n (N per palm; falls back to grip_force), lift_height, carry_dy (m,
    world +y = the humanoid's left), carry_s (carry duration). -> SkillResult (status done | no_contact | fell)."""
    from physmem.skills.base import SkillRun
    p = call.params
    if est is None:
        return SkillResult(False, "not_observed", EffortMeter(), 0.0)
    sq = float(np.clip(p.get("squeeze_n", p.get("grip_force", 30.0)), 5.0, 60.0))
    with SkillRun(ctx, "pick") as r:
        info = _transfer(ctx, est.pos, est.half_size, squeeze_n=sq, lift=float(np.clip(p.get("lift_height", 0.10), 0.03, 0.15)),
                         carry_dy=float(np.clip(p.get("carry_dy", 0.0), -0.35, 0.35)),
                         carry_T=float(np.clip(p.get("carry_s", 2.0), 0.8, 4.0)))
        st = "fell" if info["fell"] else ("done" if info["touched"] else "no_contact")
        return r.result(st == "done", st, touched=info["touched"], squeeze_n=sq)
