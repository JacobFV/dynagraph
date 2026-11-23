"""Skill runtime: context, actor-legal object estimates, the step loop and result packaging.

Skills act ONLY through the embodiment's controller (task-space trajectories, modes, gripper). They never read true
object state: everything about the world comes in as an ObjectEstimate (a pose tracker's estimate, a geometry
estimate) or as SkillCall params. What they read from the robot is proprioception (encoders, FK, wrist F/T, gripper
closure), which is actor-legal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from physmem.v2.api import EffortMeter, SkillResult


@dataclass
class ObjectEstimate:
    """actor-legal estimate of an entity. Frames: world, z up, quat wxyz."""
    pos: tuple
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    half_size: tuple = (0.03, 0.03, 0.03)      # box half extents (cylinder: (r, r, h/2))
    shape: str = 'box'                          # box | cylinder | sphere
    # articulated targets (open/close): handle pose = pos/quat; motion model
    kind: str = ''                              # '' | 'drawer' | 'door' | 'lid'
    axis: Optional[tuple] = None                # drawer: pull direction (world); door/lid: hinge axis (world)
    hinge: Optional[tuple] = None               # door/lid: a point on the hinge axis (world)
    # insertion targets: the hole (pos = hole entry centre, axis = insertion direction, e.g. (0,0,-1))
    surface_z: Optional[float] = None           # supporting surface height if known
    extra: dict = field(default_factory=dict)

    @property
    def p(self):
        return np.asarray(self.pos, float)

    @property
    def R(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(self.quat, float))
        return R.reshape(3, 3)

    @property
    def h(self):
        return np.asarray(self.half_size, float)

    def bottom_z(self):
        if self.surface_z is not None:
            return float(self.surface_z)
        R = self.R
        return float(self.p[2] - np.abs(R[2]) @ self.h)

    def top_z(self):
        return float(self.p[2] + np.abs(self.R[2]) @ self.h)


class SkillContext:
    """what a skill needs to run: model/data, the embodiment, its controller, an env hook called after every
    physics step (sensors/recording/other robots), and a time budget."""

    def __init__(self, model, data, emb, on_step: Optional[Callable] = None, before_step: Optional[Callable] = None,
                 max_time: float = 30.0, effort_weight: Optional[float] = None, record: bool = False):
        self.model, self.data, self.emb = model, data, emb
        self.ctrl = emb.controller()
        if effort_weight is not None:
            self.ctrl.effort_weight = effort_weight
        self.on_step = on_step
        self.before_step = before_step
        self.max_time = max_time
        self.record = record
        self.trace = []
        self.wall = 0.0

    def step(self, n=1):
        m, d = self.model, self.data
        t0 = time.perf_counter()
        for _ in range(n):
            if self.before_step is not None:
                self.before_step(m, d)
            self.ctrl.apply(m, d)
            mujoco.mj_step(m, d)
            if self.on_step is not None:
                self.on_step(m, d)
            if self.record:
                self.trace.append(self._snap())
        self.wall += time.perf_counter() - t0

    def _snap(self):
        out = [self.data.time]
        for e in self.emb.eff:
            p, _ = self.emb.tcp_pose(self.data, e)
            v, _ = self.emb.tcp_vel(self.model, self.data, e)
            ref = self.ctrl.reference(e)
            out.append((p.copy(), v.copy(), None if ref is None else (ref[0].copy(), ref[1].copy())))
        return out

    def run(self, until: Optional[Callable] = None, max_t: Optional[float] = None, effector=None, monitor=None):
        """step until the controller is done (time is up / guard tripped) or until() is true. NO convergence waits:
        the stop condition is time or an event. -> 'done' | 'until' | 'timeout' | 'unstable'"""
        t_end = self.data.time + (max_t if max_t is not None else self.max_time)
        while True:
            self.step()
            if monitor is not None:
                monitor(self)
            if not np.isfinite(self.data.qacc).all() or self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                return 'unstable'
            if until is not None and until(self):
                return 'until'
            if self.ctrl.done(effector):
                return 'done'
            if self.data.time >= t_end:
                return 'timeout'

    def wait(self, seconds):
        """advance a FIXED time (e.g. a gripper's close time). This is not a convergence wait."""
        n = int(round(seconds / self.model.opt.timestep))
        self.step(n)


class SkillRun:
    """context manager producing an api.SkillResult with an effort window."""

    def __init__(self, ctx: SkillContext, name: str):
        self.ctx, self.name = ctx, name
        self.log = dict(skill=name)

    def __enter__(self):
        self.meter = self.ctx.ctrl.acc.open()
        self.t0 = self.ctx.data.time
        self.w0 = self.ctx.wall
        return self

    def __exit__(self, *exc):
        self.ctx.ctrl.acc.close(self.meter)
        return False

    def result(self, ok, status, **log):
        self.log.update(log)
        dur = float(self.ctx.data.time - self.t0)
        self.log['wall_s'] = float(self.ctx.wall - self.w0)
        self.log['peaks'] = dict(self.ctx.ctrl.acc.win_peaks.get(id(self.meter), {}))
        return SkillResult(ok=bool(ok), status=status, effort=self.meter, duration_s=dur, log=self.log)
