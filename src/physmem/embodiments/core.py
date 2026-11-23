"""Embodiment layer core: robot configs, MjSpec attachment with name prefix, actuator -> torque-motor conversion,
grippers, proprioception (encoders, FK, wrist F/T).

Actuator conversion follows CommandAGI physics_run.py (same org, adapted): every actuator becomes a torque motor
(gaintype fixed, gain 1, no bias) whose ctrlrange is the joint's torque limit (forcerange / actuatorfrcrange / a
documented override). Nothing sets a position or velocity of any body; only torques move the robot.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from physmem.v2.api import EmbodimentSpec

ASSETS = Path(os.environ.get('PHYSMEM_ASSETS', str(Path.home() / 'dynagraph' / 'assets')))


def scene_defaults(spec: mujoco.MjSpec, timestep: float = 0.002):
    """physics options every v2 scene should use (child robot options are dropped by attach)."""
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.iterations = 50
    spec.option.ls_iterations = 20
    return spec


# --------------------------------------------------------------------------- configs
@dataclass
class GripperCfg:
    kind: str = 'parallel'            # parallel | dexterous | suction | flat_palm | none
    actuators: tuple = ()             # child-file actuator names
    closed: tuple = ()                # transmission length when closed, per actuator (None = from range + close_sign)
    opened: tuple = ()                # transmission length when open
    force: tuple = ()                 # max actuator effort (N or N m) used to squeeze, per actuator
    kp: float = 0.0                   # effort per unit length error (0 = auto: saturate at 10% stroke)
    kd_ratio: float = 0.05            # kd = kd_ratio * kp
    pad_bodies: tuple = ()            # bodies whose contacts count as grasp contacts (child names)
    close_time: float = 0.4           # free closing time; sets the drive's implicit damping (gearbox/back-EMF model)
    max_width: float = 0.08           # max object width between the pads (m)
    tip: float = 0.012                # fingertip extent beyond the TCP along the approach, CLOSED (m, scripts/robots_tips.py)
    link_h: float = 0.055             # height above the TCP where the gripper widens (linkage / palm) (m)
    pad_per_drive: float = 1.0        # pad normal force per side (N) per unit drive effort, measured (scripts/robots_grip_calib.py)
    grip_n: float = 0.0               # nominal squeeze per side (N); 0 = use `force`. make(name, grip_force=N) overrides
    pad_friction: float = 0.0         # sliding friction of the pad geoms (0 = keep the model's)
    jaw_dir: int = 0                  # 0: symmetric fingers; +-1: single moving jaw on the +-TCP-y side of a fixed jaw


@dataclass
class EffectorCfg:
    name: str
    joints: tuple                     # controlled joint names (child names), base -> tip
    site_body: str                    # body carrying the tool centre point
    tcp_pos: tuple = (0, 0, 0)        # TCP in that body's frame
    tcp_quat: tuple = None            # TCP frame (default: from approach/closing axes below)
    approach: tuple = (0, 0, 1)       # body-frame direction out of the gripper (TCP z)
    closing: tuple = (0, 1, 0)        # body-frame finger closing axis (TCP y)
    home: tuple = ()                  # joint angles
    gripper: GripperCfg = field(default_factory=GripperCfg)
    rot_weight: float = 1.0           # orientation task weight (<1 for under-actuated arms)
    ft_bodies: tuple = ()             # extra bodies counted by the wrist F/T (default: subtree of site_body's wrist)
    wrist_body: str = ''              # body whose subtree the wrist F/T measures (default site_body)


@dataclass
class RobotCfg:
    name: str
    kind: str                         # arm | bimanual | humanoid | mobile_manipulator
    xml: str                          # path relative to ASSETS
    effectors: tuple
    attach: tuple = ()                # ((parent_site_or_body, child_xml, child_prefix), ...) e.g. a gripper on a flange
    base_joints: tuple = ()           # mobile base joints (x, y, yaw) controlled by the base servo
    torque_override: dict = field(default_factory=dict)   # child actuator name -> (lo, hi)
    reach_m: float = 0.8
    payload_kg: float = 3.0
    max_ee_speed: float = 1.0         # measured (scripts/robots_gate.py); filled from docs/ROBOTS.md table
    max_ee_force: float = 50.0
    limits: dict = field(default_factory=lambda: dict(v=1.0, a=8.0, j=150.0, w=3.0))  # retiming budget
    gains: dict = field(default_factory=dict)
    floating_base: bool = False
    source: str = ''
    drop_bodies: tuple = ()           # extra top-level bodies to delete (e.g. a floor shipped with the model)
    joint_overrides: dict = field(default_factory=dict)   # joint -> {armature, damping, frictionloss} (documented)
    tool: dict = field(default_factory=dict)   # e.g. dict(kind='suction', site='attachment_site', stem=0.07, radius=0.02, gain=120)
    front_yaw: float = 0.0            # yaw of the model's working direction; build() rotates it onto mount +x
    home_target: tuple = (0.5, 0.0, 0.3)  # nominal home TCP point in the mount frame (TCP pointing down)
    base_body: str = ''               # name of root body (for mount)


# --------------------------------------------------------------------------- helpers
_OPT_FIELDS = ('timestep', 'integrator', 'cone', 'impratio', 'iterations', 'ls_iterations', 'noslip_iterations',
               'solver', 'jacobian', 'tolerance')


def _match_options(child, parent):
    """copy the parent's physics options into a child before attach (they are dropped anyway; avoids conflict spam)."""
    for f in _OPT_FIELDS:
        try:
            setattr(child.option, f, getattr(parent.option, f))
        except Exception:
            pass


def torque_motorize(child: mujoco.MjSpec, override: dict, model_for_limits: mujoco.MjModel):
    """convert every actuator of `child` into a torque motor with ctrlrange = torque limit. returns {name: (lo,hi)}."""
    lims = {}
    m = model_for_limits
    for i, a in enumerate(child.actuators):
        name = a.name
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_BODY:   # adhesion (suction) stays an adhesion actuator, ctrl 0..1
            lims[name] = (0.0, 1.0)
            continue
        lo, hi = None, None
        if name in override:
            lo, hi = override[name]
        else:
            fr = m.actuator_forcerange[i]
            if m.actuator_forcelimited[i] and fr[1] > fr[0]:
                lo, hi = fr
            elif m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                jr = m.jnt_actfrcrange[m.actuator_trnid[i, 0]]
                if jr[1] > jr[0]:
                    lo, hi = jr
        if lo is None:
            raise ValueError(f'actuator {name}: no torque limit; add to torque_override')
        g = float(m.actuator_gear[i, 0]) or 1.0
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.inheritrange = 0
        a.gainprm[:] = 0
        a.gainprm[0] = 1
        a.biasprm[:] = 0
        a.ctrlrange = [lo / g, hi / g]
        a.ctrllimited = 1
        a.forcerange = [lo, hi]
        a.forcelimited = 1
        lims[name] = (float(lo), float(hi))
    return lims


def add_suction_cup(child, site='attachment_site', stem=0.07, radius=0.02, gain=120.0, mass=0.25, **_):
    """vacuum cup tool: a stem + a compliant cup pad on the flange site, and an ADHESION actuator on the cup body
    (MuJoCo adhesion: attractive force at the cup's contacts, ctrl 0..1 x gain N in total). The controller turns it on
    only while the cup is sealed (see osc.py). No welds. gain 120 N ~ a 40 mm cup at -0.95 bar."""
    s = child.site(site)
    parent = s.parent
    b = parent.add_body(name='cup', pos=list(s.pos), quat=list(s.quat))
    b.add_geom(name='cup_stem', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, stem / 2, 0], pos=[0, 0, stem / 2],
               mass=mass * 0.6, rgba=[0.3, 0.3, 0.32, 1])
    b.add_geom(name='cup_pad', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.004, 0], pos=[0, 0, stem + 0.004],
               mass=mass * 0.4, rgba=[0.15, 0.15, 0.15, 1], friction=[1.2, 0.01, 0.001], condim=4,
               margin=0.004, gap=0.004, solref=[0.01, 1], solimp=[0.9, 0.95, 0.002, 0.5, 2])
    a = child.add_actuator(name='suction', trntype=mujoco.mjtTrn.mjTRN_BODY, target='cup')
    a.gaintype, a.biastype, a.dyntype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtDyn.mjDYN_NONE
    a.gainprm[0] = gain
    a.ctrllimited = 1
    a.ctrlrange = [0, 1]
    return b


def tcp_quat(e):
    if e.tcp_quat is not None:
        return np.asarray(e.tcp_quat, float)
    z = np.asarray(e.approach, float); z /= np.linalg.norm(z)
    y = np.asarray(e.closing, float); y = y - z * (y @ z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).reshape(9))
    return q


def _subtree(m, root):
    out = {root}
    for b in range(root + 1, m.nbody):
        p = b
        while p > 0 and p not in out:
            p = m.body_parentid[p]
        if p in out:
            out.add(b)
    return out


def _actuator_length_range(m, i):
    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
        return tuple(m.jnt_range[m.actuator_trnid[i, 0]])
    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
        t = m.actuator_trnid[i, 0]
        lo = hi = 0.0
        for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
            j, c = m.wrap_objid[w], m.wrap_prm[w]
            a, b = c * m.jnt_range[j][0], c * m.jnt_range[j][1]
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi
    raise ValueError('unsupported transmission')


# --------------------------------------------------------------------------- the embodiment
class MjEmbodiment:
    """api.Embodiment for any fixed-base / mobile-base robot described by a RobotCfg."""

    def __init__(self, cfg: RobotCfg, prefix: str = 'r0'):
        self.cfg = cfg
        self.prefix = prefix.rstrip('/') + '/'
        self.spec = EmbodimentSpec(name=cfg.name, kind=cfg.kind, effectors=tuple(e.name for e in cfg.effectors),
                                   gripper=cfg.effectors[0].gripper.kind, reach_m=cfg.reach_m,
                                   payload_kg=cfg.payload_kg, max_ee_speed=cfg.max_ee_speed,
                                   max_ee_force=cfg.max_ee_force, dof=0, floating_base=cfg.floating_base,
                                   source=cfg.source)
        self.torque_limits = {}
        self._ctrl = None
        self.model = None

    # ---------------------------------------------------------------- build
    def load_child(self):
        cfg = self.cfg
        child = mujoco.MjSpec.from_file(str(ASSETS / cfg.xml))
        for site, xml, pre in cfg.attach:
            g = mujoco.MjSpec.from_file(str(ASSETS / xml))
            for k in list(g.keys):
                g.delete(k)
            s = child.site(site)
            if s is None:  # a body name: add a site at its origin
                s = child.body(site).add_site(name=f'{site}_attach')
            _match_options(g, child)
            child.attach(g, site=s, prefix=pre)
        for k in list(child.keys):
            child.delete(k)
        for n in cfg.drop_bodies:
            b = child.body(n)
            if b is not None:
                child.delete(b)
        # drop lights/cameras-free: keep; remove ground planes shipped with a model
        for gm in list(child.geoms):
            if gm.type == mujoco.mjtGeom.mjGEOM_PLANE:
                child.delete(gm)
        if cfg.tool.get('kind') == 'suction':
            add_suction_cup(child, **cfg.tool)
        for jn, kv in cfg.joint_overrides.items():
            j = child.joint(jn)
            for k, v in kv.items():
                setattr(j, k, v)
        # TCP sites
        for e in cfg.effectors:
            b = child.body(e.site_body)
            assert b is not None, f'{cfg.name}: no body {e.site_body}'
            b.add_site(name=f'tcp_{e.name}', pos=list(e.tcp_pos), quat=list(tcp_quat(e)), size=[0.006, 0, 0],
                       rgba=[1, 0.2, 0.2, 0.5], group=4)
        for e in cfg.effectors:
            if e.gripper.pad_friction:
                for bn in e.gripper.pad_bodies:
                    b = child.body(bn)
                    for gm in (b.geoms if b is not None else []):
                        gm.friction = [e.gripper.pad_friction, 0.005, 0.0001]
                        gm.condim = 4
        lm = child.compile()
        # finger drives: implicit damping so a free close takes ~close_time at full force (stable at any gain)
        for e in cfg.effectors:
            gr = e.gripper
            for n_, an in enumerate(gr.actuators if gr.kind != 'suction' else ()):
                i = lm.actuator(an).id
                lo, hi = _actuator_length_range(lm, i)
                stroke = abs((gr.closed[n_] if gr.closed else hi) - (gr.opened[n_] if gr.opened else lo))
                F = gr.grip_n / gr.pad_per_drive if gr.grip_n else (gr.force[n_] if gr.force else float(lm.actuator_forcerange[i][1]))
                b = F / max(stroke / gr.close_time, 1e-6)
                if lm.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                    js = [lm.actuator_trnid[i, 0]]
                else:
                    t = lm.actuator_trnid[i, 0]
                    js = [lm.wrap_objid[w] for w in range(lm.tendon_adr[t], lm.tendon_adr[t] + lm.tendon_num[t])]
                for j in js:
                    cj = child.joint(lm.joint(j).name)
                    cj.damping = np.maximum(np.asarray(cj.damping, float), b / len(js))
        lm = child.compile()
        self.torque_limits = torque_motorize(child, cfg.torque_override, lm)
        return child

    def build(self, mjspec: mujoco.MjSpec, mount_pose=((0, 0, 0), (1, 0, 0, 0))) -> None:
        child = self.load_child()
        pos, quat = mount_pose
        qy = np.array([np.cos(-self.cfg.front_yaw / 2), 0, 0, np.sin(-self.cfg.front_yaw / 2)])
        qf = np.zeros(4)
        mujoco.mju_mulQuat(qf, np.asarray(quat, float), qy)
        fr = mjspec.worldbody.add_frame(pos=list(map(float, pos)), quat=list(map(float, qf)))
        _match_options(child, mjspec)
        mjspec.attach(child, prefix=self.prefix, frame=fr)
        self.mount_pose = (np.asarray(pos, float), np.asarray(quat, float))

    # ---------------------------------------------------------------- bind
    def bind(self, model, data) -> None:
        m, P = model, self.prefix
        self.model, self.data = model, data
        self.act_ids = np.array([i for i in range(m.nu) if m.actuator(i).name.startswith(P)], int)
        self.act_lo = m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.act_hi = m.actuator_ctrlrange[self.act_ids, 1].copy()
        self.act_gear = m.actuator_gear[self.act_ids, 0].copy()
        self.dof_of_act = {}
        for k, i in enumerate(self.act_ids):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                self.dof_of_act[k] = int(m.jnt_dofadr[m.actuator_trnid[i, 0]])
        self.eff = {}
        all_dofs = []
        for e in self.cfg.effectors:
            jids = [m.joint(P + j).id for j in e.joints]
            dofs = np.array([m.jnt_dofadr[j] for j in jids], int)
            qadr = np.array([m.jnt_qposadr[j] for j in jids], int)
            acts = []
            for j in jids:
                a = [k for k, i in enumerate(self.act_ids) if m.actuator_trntype[i] == 0 and m.actuator_trnid[i, 0] == j]
                assert a, f'joint {m.joint(j).name} has no actuator'
                acts.append(a[0])
            site = m.site(P + f'tcp_{e.name}').id
            wb = m.body(P + (e.wrist_body or e.site_body)).id
            ft_bodies = _subtree(m, wb) | {m.body(P + b).id for b in e.ft_bodies}
            ft_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in ft_bodies], int)
            gr = e.gripper
            gacts = [int(np.where(self.act_ids == m.actuator(P + a).id)[0][0]) for a in gr.actuators]
            gopen, gclosed = [], []
            for n_, k in enumerate(gacts):
                if gr.kind == 'suction':
                    gopen.append(0.0); gclosed.append(1.0)
                    continue
                lo, hi = _actuator_length_range(m, self.act_ids[k])
                gopen.append(gr.opened[n_] if gr.opened else lo)
                gclosed.append(gr.closed[n_] if gr.closed else hi)
            pad = {m.body(P + b).id for b in gr.pad_bodies}
            lim = np.array([[self.act_lo[a], self.act_hi[a]] for a in acts]) * self.act_gear[acts][:, None]
            if gr.kind == 'suction':
                gforce = np.ones(len(gacts))
            elif gr.grip_n:
                gforce = np.array([min(gr.grip_n / gr.pad_per_drive, self.act_hi[a] * self.act_gear[a]) for a in gacts], float)
            else:
                gforce = np.array(gr.force if gr.force else [self.act_hi[a] * self.act_gear[a] for a in gacts], float)
            self.eff[e.name] = dict(cfg=e, dofs=dofs, qadr=qadr, acts=np.array(acts, int), site=site,
                                    body=m.site_bodyid[site], ft_bodies=ft_bodies, ft_geoms=ft_geoms,
                                    gacts=np.array(gacts, int), gopen=np.array(gopen), gclosed=np.array(gclosed),
                                    gforce=gforce, gforce_travel=gforce.copy(),
                                    pad=pad, tau_lim=lim, home=np.array(e.home, float))
            all_dofs += list(dofs)
        # base
        self.base = None
        if self.cfg.base_joints:
            jids = [m.joint(P + j).id for j in self.cfg.base_joints]
            dofs = np.array([m.jnt_dofadr[j] for j in jids], int)
            acts = [int(np.where((m.actuator_trnid[self.act_ids, 0] == j) & (m.actuator_trntype[self.act_ids] == 0))[0][0]) for j in jids]
            self.base = dict(dofs=dofs, qadr=np.array([m.jnt_qposadr[j] for j in jids], int), acts=np.array(acts),
                             tau_lim=np.array([[self.act_lo[a], self.act_hi[a]] for a in acts]))
        roots = {int(m.body_rootid[m.site_bodyid[e['site']]]) for e in self.eff.values()}
        if self.cfg.base_body:
            roots.add(int(m.body_rootid[m.body(P + self.cfg.base_body).id]))
        self.root_body = min(roots)
        self.bodies = set()
        for r in roots:
            self.bodies |= _subtree(m, r)
        self.geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bodies], int)
        self.dofs = np.array(sorted(set(int(d) for d in all_dofs)), int)
        object.__setattr__(self.spec, 'dof', int(len(self.dofs) + (len(self.base['dofs']) if self.base else 0)))
        self._ctrl = None

    def reset(self, model, data, home=True):
        """put the robot at its home posture (call once after compile, before the first step)."""
        for e in self.eff.values():
            if len(e['home']):
                data.qpos[e['qadr']] = e['home']
            data.qvel[e['dofs']] = 0
            # start with the gripper open (menagerie qpos0 is often closed)
            for k, a in enumerate(e['gacts']):
                i = self.act_ids[a]
                if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                    j = model.actuator_trnid[i, 0]
                    data.qpos[model.jnt_qposadr[j]] = e['gopen'][k]
                elif model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
                    t = model.actuator_trnid[i, 0]
                    ws = range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t])
                    csum = sum(model.wrap_prm[w] for w in ws)
                    for w in ws:
                        data.qpos[model.jnt_qposadr[model.wrap_objid[w]]] = e['gopen'][k] / csum
            if len(e['gacts']) and self._ctrl is not None:
                self._ctrl.st.get(next(iter(self.eff)), {})
        data.ctrl[self.act_ids] = 0
        mujoco.mj_forward(model, data)
        if self._ctrl is not None:
            self._ctrl.reset(model, data)

    # ---------------------------------------------------------------- sensing (actor-legal)
    def tcp_pose(self, data, eff):
        e = self.eff[eff]
        p = data.site_xpos[e['site']].copy()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[e['site']])
        return p, q

    def tcp_vel(self, model, data, eff):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.eff[eff]['site'], v, 0)
        return v[3:], v[:3]

    def wrist_ft(self, model, data, eff):
        """contact wrench on the effector (gripper + wrist subtree) from anything that is not the robot itself,
        world frame, about the TCP: what an ideal gravity-compensated wrist F/T sensor reads."""
        e = self.eff[eff]
        F, Tq = np.zeros(3), np.zeros(3)
        if data.ncon == 0:
            return F, Tq, 0
        g1, g2 = data.contact.geom1, data.contact.geom2
        in1, in2 = np.isin(g1, e['ft_geoms']), np.isin(g2, e['ft_geoms'])
        own1, own2 = np.isin(g1, self.geoms), np.isin(g2, self.geoms)
        sel = np.where((in1 & ~own2) | (in2 & ~own1))[0]
        f6 = np.zeros(6)
        p0 = data.site_xpos[e['site']]
        for k in sel:
            mujoco.mj_contactForce(model, data, int(k), f6)
            fw = data.contact.frame[k].reshape(3, 3).T @ f6[:3]  # force on geom2
            if in1[k]:
                fw = -fw
            F += fw
            Tq += np.cross(data.contact.pos[k] - p0, fw)
        return F, Tq, len(sel)

    def gripper_state(self, model, data, eff):
        """(normalized closure 0=open..1=closed, finger contact count with non-robot geoms)."""
        e = self.eff[eff]
        if not len(e['gacts']):
            return 0.0, 0
        if e['cfg'].gripper.kind == 'suction':
            sealed, n = self.cup_seal(model, data, eff, max_tilt_deg=45.0)
            return float(data.ctrl[self.act_ids[e['gacts'][0]]] > 0.05 and sealed), n
        L = data.actuator_length[self.act_ids[e['gacts']]]
        c = float(np.mean((L - e['gopen']) / (e['gclosed'] - e['gopen'] + 1e-12)))
        n = 0
        if data.ncon and e['pad']:
            pg = np.array([g for g in e['ft_geoms'] if model.geom_bodyid[g] in e['pad']], int)
            g1, g2 = data.contact.geom1, data.contact.geom2
            n = int(np.sum((np.isin(g1, pg) & ~np.isin(g2, self.geoms)) | (np.isin(g2, pg) & ~np.isin(g1, self.geoms))))
        return c, n

    def set_grip_force(self, eff, newtons):
        """per-pad squeeze (N), via the measured GripperCfg.pad_per_drive; clipped to the drive limit. Returns the
        force actually set (N)."""
        e = self.eff[eff]
        gr = e['cfg'].gripper
        if gr.kind == 'suction':
            return 0.0
        lim = np.array([self.act_hi[a] * self.act_gear[a] for a in e['gacts']])
        e['gforce'] = np.minimum(float(newtons) / gr.pad_per_drive, lim)
        return float(e['gforce'][0] * gr.pad_per_drive)

    def max_grip_force(self, eff):
        e = self.eff[eff]
        gr = e['cfg'].gripper
        return 0.0 if gr.kind == 'suction' else float(self.act_hi[e['gacts'][0]] * self.act_gear[e['gacts'][0]] * gr.pad_per_drive)

    def cup_seal(self, model, data, eff, max_tilt_deg=20.0):
        """vacuum-cup seal sensing (a pressure switch in reality): the cup pad touches a non-robot surface with the
        contact normal within max_tilt of the cup axis. -> (sealed, n_contacts)"""
        e = self.eff[eff]
        cg = e.get('_cupg')
        if cg is None:
            cg = e['_cupg'] = model.geom(self.prefix + 'cup_pad').id
        if not data.ncon:
            return False, 0
        z = data.site_xmat[e['site']].reshape(3, 3)[:, 2]
        n, ok = 0, 0
        for k in np.where((data.contact.geom1 == cg) | (data.contact.geom2 == cg))[0]:
            other = data.contact.geom2[k] if data.contact.geom1[k] == cg else data.contact.geom1[k]
            if other in self.geoms or data.contact.dist[k] > 0.004:
                continue
            n += 1
            nrm = data.contact.frame[k][:3]
            ok += abs(float(nrm @ z)) > np.cos(np.radians(max_tilt_deg))
        return (n > 0 and ok == n), n

    def proprio(self, data) -> dict:
        m = self.model
        out = {}
        for name, e in self.eff.items():
            p, q = self.tcp_pose(data, name)
            v, w = self.tcp_vel(m, data, name)
            F, T, _ = self.wrist_ft(m, data, name)
            c, n = self.gripper_state(m, data, name)
            out[name] = dict(q=data.qpos[e['qadr']].copy(), dq=data.qvel[e['dofs']].copy(), ee_pos=p, ee_quat=q,
                             ee_vel=v, ee_angvel=w, wrist_force=F, wrist_torque=T, grip=c, grip_contacts=n)
        if self.base is not None:
            out['base'] = dict(q=data.qpos[self.base['qadr']].copy(), dq=data.qvel[self.base['dofs']].copy())
        return out

    def controller(self):
        if self._ctrl is None:
            from physmem.embodiments.osc import TaskSpaceController
            self._ctrl = TaskSpaceController(self)
            if self.model is not None:
                self._ctrl.reset(self.model, self.data)
        return self._ctrl
