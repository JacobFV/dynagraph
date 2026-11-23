"""Unitree G1 (29 dof) + Dex3 hands: floating-base humanoid with balance by a whole-body QP (wbc_qp.py, adapted from
CommandAGI wbc.py), standing tasks adapted from CommandAGI physics_run.py Robot.control (non-walking part): CoM over
the feet, root height, root + torso upright, hands along min-jerk task trajectories (pos/vel/acc feedforward + bounded
integral), posture. Fingers are outside the QP and get their own torque-limited joint PD.

`g1_dex3_fixed` is a clearly labelled FALLBACK: the pelvis free joint is removed (pelvis welded to the mount frame at
standing height); same controller with no balance tasks and no contact forces.

Duck-types core.MjEmbodiment / osc.TaskSpaceController so skills work unchanged.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from physmem.v2.api import EffortMeter, EmbodimentSpec, TaskTrajectory
from physmem.embodiments.core import ASSETS, EffectorCfg, GripperCfg, RobotCfg, _subtree, tcp_quat, torque_motorize
from physmem.embodiments.osc import EffortAccumulator, clip_norm
from physmem.embodiments.trajectory import Plan, quat_err
from physmem.embodiments import wbc_qp

G1_XML = 'menagerie/unitree_g1/g1_with_hands.xml'
SIDES = ('left', 'right')
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')
# stand posture (legs bent) and a ready arm posture with the hands in front of the torso
STAND = {'hip_pitch': -0.3, 'knee': 0.6, 'ankle_pitch': -0.3}
ARM_HOME = {'left': (0.2, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0), 'right': (0.2, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0)}
# Dex3 power grasp (left; right mirrors the signs of thumb_1/2 and fingers)
GRASP_L = dict(thumb_0=0.0, thumb_1=0.6, thumb_2=1.2, middle_0=-1.2, middle_1=-1.3, index_0=-1.2, index_1=-1.3)
OPEN_L = dict(thumb_0=0.0, thumb_1=-0.3, thumb_2=0.0, middle_0=0.0, middle_1=0.0, index_0=0.0, index_1=0.0)

G1_GAINS = dict(com_kp=50.0, root_z_kp=100.0, root_kp=200.0, torso_kp=100.0, hand_kp=400.0, hand_ki=1500.0,
                hand_i_max=0.004, hand_e_max=0.05, hand_kr=200.0, hand_w=3000.0, hand_wr=300.0, post_kp=40.0,
                post_w=0.2, imp_k=300.0, imp_kr=10.0, guard_n=15.0, force_kd=30.0, force_ki=4.0,
                finger_kp=6.0, finger_kd=0.15, balance_w=1000.0, root_w=300.0, torso_w=30.0,
                f_cap=15.0, f_back=0.3, ext_total=80.0, com_shift=0.06)


def _hand_val(side, j, table):
    v = table[j]
    return v if side == 'left' or j == 'thumb_0' else -v


def _cfg(fixed):
    effs = []
    for s in SIDES:
        sg = -1 if s == 'left' else 1
        effs.append(EffectorCfg(
            name=s, joints=tuple(f'{s}_{j}_joint' for j in ARM), site_body=f'{s}_wrist_yaw_link',
            tcp_pos=(0.11, 0.035 * sg, 0.0), approach=(0, sg, 0), closing=(1, 0, 0), home=ARM_HOME[s],
            rot_weight=0.3,
            gripper=GripperCfg(kind='dexterous', actuators=tuple(f'{s}_hand_{j}_joint' for j in HAND),
                               pad_bodies=(f'{s}_wrist_yaw_link',) + tuple(f'{s}_hand_{j}_link' for j in HAND))))
    return RobotCfg(name='g1_dex3_fixed' if fixed else 'g1_dex3', kind='humanoid', xml=G1_XML, effectors=tuple(effs),
                    reach_m=0.55, payload_kg=1.0, max_ee_speed=0.6, max_ee_force=20.0,
                    limits=dict(v=0.8, a=6.0, j=150.0, w=3.0), gains=dict(G1_GAINS), floating_base=not fixed,
                    source='mujoco_menagerie@f3b6013 unitree_g1/g1_with_hands.xml (BSD-3-Clause)', base_body='pelvis')


class G1Embodiment:
    """api.Embodiment for the G1 + Dex3 (floating base, or fixed-base fallback)."""

    def __init__(self, prefix: str = 'r0', fixed_base: bool = False):
        self.fixed = fixed_base
        self.cfg = _cfg(fixed_base)
        self.prefix = prefix.rstrip('/') + '/'
        c = self.cfg
        self.spec = EmbodimentSpec(name=c.name, kind='humanoid', effectors=SIDES, gripper='dexterous',
                                   reach_m=c.reach_m, payload_kg=c.payload_kg, max_ee_speed=c.max_ee_speed,
                                   max_ee_force=c.max_ee_force, dof=29 + 14, floating_base=not fixed_base,
                                   source=c.source + (' [FIXED-BASE FALLBACK: pelvis welded]' if fixed_base else ''))
        self.torque_limits = {}
        self._ctrl = None
        self.model = None
        self.base = None

    # ---------------------------------------------------------------- build
    def _posture(self):
        q = {}
        for s in SIDES:
            for j, v in STAND.items():
                q[f'{s}_{j}_joint'] = v
            for j, v in zip(ARM, ARM_HOME[s]):
                q[f'{s}_{j}_joint'] = v
            for j in HAND:
                q[f'{s}_hand_{j}_joint'] = _hand_val(s, j, OPEN_L)
        return q

    def load_child(self):
        child = mujoco.MjSpec.from_file(str(ASSETS / self.cfg.xml))
        for k in list(child.keys):
            child.delete(k)
        for gm in list(child.geoms):
            if gm.type == mujoco.mjtGeom.mjGEOM_PLANE:
                child.delete(gm)
        # no self-collision (as CommandAGI physics_run): robot geoms contype bit 1, touch the world (bit 0) only
        for gm in child.geoms:
            if gm.contype or gm.conaffinity:
                gm.contype, gm.conaffinity = 2, 1
        for e in self.cfg.effectors:
            child.body(e.site_body).add_site(name=f'tcp_{e.name}', pos=list(e.tcp_pos), quat=list(tcp_quat(e)),
                                             size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 0.5], group=4)
            for a in e.gripper.actuators:  # finger drives: implicit damping (gearbox), stable at 2 ms
                j = child.joint(a)
                j.damping = np.maximum(np.asarray(j.damping, float), 0.05)
        # standing pelvis height above the soles for the stand posture
        lm = child.compile()
        ld = mujoco.MjData(lm)
        for n, v in self._posture().items():
            ld.qpos[lm.jnt_qposadr[lm.joint(n).id]] = v
        mujoco.mj_kinematics(lm, ld)
        pel = lm.body('pelvis').id
        feet = [g for g in range(lm.ngeom) if lm.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                and lm.body(lm.geom_bodyid[g]).name.endswith('ankle_roll_link')]
        low = min(ld.geom_xpos[g][2] - lm.geom_size[g][0] for g in feet)
        self.stand_height = float(ld.xpos[pel][2] - low)
        if self.fixed:
            fj = [j for j in child.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
            for j in fj:
                child.delete(j)
            child.body('pelvis').pos = [0, 0, 0]
        else:
            child.body('pelvis').pos = [0, 0, 0]
        lm = child.compile()
        self.torque_limits = torque_motorize(child, {}, lm)
        return child

    def build(self, mjspec: mujoco.MjSpec, mount_pose=((0, 0, 0), (1, 0, 0, 0))) -> None:
        child = self.load_child()
        pos, quat = mount_pose
        p = np.asarray(pos, float) + np.array([0, 0, self.stand_height])
        fr = mjspec.worldbody.add_frame(pos=list(map(float, p)), quat=list(map(float, quat)))
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
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in self.act_ids], int)
        self.act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in self.act_ids], int)
        self.root_body = m.body(P + 'pelvis').id
        self.bodies = _subtree(m, self.root_body)
        self.geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bodies], int)
        self.free = None
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == self.root_body:
                self.free = j
        act_of_joint = {int(m.actuator_trnid[i, 0]): k for k, i in enumerate(self.act_ids)}
        self.eff = {}
        hand_dofs = []
        for e in self.cfg.effectors:
            jids = [m.joint(P + j).id for j in e.joints]
            dofs = np.array([m.jnt_dofadr[j] for j in jids], int)
            qadr = np.array([m.jnt_qposadr[j] for j in jids], int)
            acts = np.array([act_of_joint[j] for j in jids], int)
            site = m.site(P + f'tcp_{e.name}').id
            wb = m.body(P + e.site_body).id
            ft_bodies = _subtree(m, wb)
            ft_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in ft_bodies], int)
            gj = [m.joint(P + a).id for a in e.gripper.actuators]
            gacts = np.array([act_of_joint[j] for j in gj], int)
            s = e.name
            gopen = np.array([_hand_val(s, j, OPEN_L) for j in HAND])
            gclosed = np.array([_hand_val(s, j, GRASP_L) for j in HAND])
            hand_dofs += [int(m.jnt_dofadr[j]) for j in gj]
            self.eff[s] = dict(cfg=e, dofs=dofs, qadr=qadr, acts=acts, site=site, body=int(m.site_bodyid[site]),
                               ft_bodies=ft_bodies, ft_geoms=ft_geoms, gacts=gacts, gopen=gopen, gclosed=gclosed,
                               gqadr=np.array([m.jnt_qposadr[j] for j in gj], int),
                               gdofs=np.array([m.jnt_dofadr[j] for j in gj], int),
                               gforce=self.act_hi[gacts] * self.act_gear[gacts],
                               pad={m.body(P + b).id for b in e.gripper.pad_bodies},
                               tau_lim=np.column_stack([self.act_lo[acts], self.act_hi[acts]]) * self.act_gear[acts][:, None],
                               home=np.array(e.home, float))
        self.hand_dofs = set(hand_dofs)
        self.dofs = np.array(sorted(int(d) for d in self.act_dof), int)
        self.feet_bodies = [m.body(P + f'{s}_ankle_roll_link').id for s in SIDES]
        self._ctrl = None

    def reset(self, model, data, home=True):
        m, P = model, self.prefix
        for n, v in self._posture().items():
            data.qpos[m.jnt_qposadr[m.joint(P + n).id]] = v
        if self.free is not None:
            a = m.jnt_qposadr[self.free]
            pos, quat = self.mount_pose
            data.qpos[a:a + 3] = pos + np.array([0, 0, self.stand_height])
            data.qpos[a + 3:a + 7] = quat
            data.qvel[m.jnt_dofadr[self.free]:m.jnt_dofadr[self.free] + 6] = 0
        data.qvel[self.act_dof] = 0
        data.ctrl[self.act_ids] = 0
        mujoco.mj_forward(model, data)
        if self._ctrl is not None:
            self._ctrl.reset(model, data)

    # ---------------------------------------------------------------- sensing
    def tcp_pose(self, data, eff):
        e = self.eff[eff]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[e['site']])
        return data.site_xpos[e['site']].copy(), q

    def tcp_vel(self, model, data, eff):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.eff[eff]['site'], v, 0)
        return v[3:], v[:3]

    def wrist_ft(self, model, data, eff):
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
            fw = data.contact.frame[k].reshape(3, 3).T @ f6[:3]
            if in1[k]:
                fw = -fw
            F += fw
            Tq += np.cross(data.contact.pos[k] - p0, fw)
        return F, Tq, len(sel)

    def gripper_state(self, model, data, eff):
        e = self.eff[eff]
        q = data.qpos[e['gqadr']]
        st = e['gclosed'] - e['gopen']
        mask = np.abs(st) > 1e-6
        c = float(np.mean((q[mask] - e['gopen'][mask]) / st[mask]))
        n = 0
        if data.ncon:
            pg = np.array([g for g in e['ft_geoms'] if model.geom_bodyid[g] in e['pad']], int)
            g1, g2 = data.contact.geom1, data.contact.geom2
            n = int(np.sum((np.isin(g1, pg) & ~np.isin(g2, self.geoms)) | (np.isin(g2, pg) & ~np.isin(g1, self.geoms))))
        return c, n

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
        rb = self.root_body
        out['base'] = dict(pos=data.xpos[rb].copy(), quat=data.xquat[rb].copy(),
                           q=data.qpos[self.act_qadr].copy(), dq=data.qvel[self.act_dof].copy())
        return out

    def controller(self):
        if self._ctrl is None:
            self._ctrl = WholeBodyController(self)
            if self.model is not None:
                self._ctrl.reset(self.model, self.data)
        return self._ctrl


class WholeBodyController:
    """api.Controller for G1Embodiment: one ProxQP whole-body solve per control tick."""

    def __init__(self, emb: G1Embodiment, effort_weight: float = 0.0):
        self.emb = emb
        self.g = dict(G1_GAINS)
        self.g.update(emb.cfg.gains)
        self.limits = dict(emb.cfg.limits)
        self.effort_weight = effort_weight
        self.acc = EffortAccumulator()
        self.t = 0.0
        self.st = {}
        self.qp_status = {}
        self._ready = False

    @property
    def effort(self):
        return self.acc.total

    # ------------------------------------------------------------------ lifecycle
    def reset(self, model, data):
        emb = self.emb
        self.m, self.d = model, data
        self.dt = model.opt.timestep
        self.t = float(data.time)
        P = emb.prefix
        body_dofs = [int(d) for d in emb.act_dof if int(d) not in emb.hand_dofs]
        if emb.free is not None:
            v0 = int(model.jnt_dofadr[emb.free])
            dofs = list(range(v0, v0 + 6)) + body_dofs
        else:
            dofs = body_dofs
        lims = {}
        actuated = []
        for k, dof in enumerate(emb.act_dof):
            if int(dof) in emb.hand_dofs:
                continue
            actuated.append(int(dof))
            lims[int(dof)] = (emb.act_lo[k] * emb.act_gear[k], emb.act_hi[k] * emb.act_gear[k])
        self.wb = wbc_qp.WholeBody(model, data, emb.root_body, dofs, actuated, lims, mu=0.6,
                                   floating=emb.free is not None)
        self.feet = [wbc_qp.Foot(model, b) for b in emb.feet_bodies] if emb.free is not None else []
        self.k_of_dof = {int(d): k for k, d in enumerate(emb.act_dof)}
        self.arm_dofs = {int(x) for e in emb.eff.values() for x in e['dofs']}
        self.arm_dofs |= {int(model.jnt_dofadr[model.joint(P + f'waist_{a}_joint').id]) for a in ('yaw', 'roll', 'pitch')}
        # posture: every QP joint toward its reset value
        self.posture = {int(d): float(data.qpos[a]) for d, a in zip(emb.act_dof, emb.act_qadr)}
        self.root_z0 = float(data.xpos[emb.root_body][2])
        self.root_q0 = data.xquat[emb.root_body].copy()
        self.torso = model.body(P + 'torso_link').id
        self.com_off = np.zeros(2)
        self.prev_Jcom = None
        self.st = {}
        for name, e in emb.eff.items():
            p, q = emb.tcp_pose(data, name)
            self.st[name] = dict(plans=[], hold=(p, q), mode='track', kw={}, ei=np.zeros(3), grip=0.0,
                                 tripped=False, fi=0.0, touched=False, ft=np.zeros(3), ref=(p, np.zeros(3), q),
                                 err=0.0, alpha=1.0)
        self.tau_prev = np.zeros(len(emb.act_ids))
        self.qd1 = data.qvel.copy()
        self.qd2 = data.qvel.copy()
        self.tick = 0
        self.weight = float(sum(model.body_mass[b] for b in emb.bodies)) * 9.81
        self._ready = True

    def _ensure(self, model, data):
        if not self._ready or self.m is not model:
            self.reset(model, data)

    # ------------------------------------------------------------------ api (same as TaskSpaceController)
    def command(self, traj: TaskTrajectory, preempt: bool = False) -> dict:
        s = self.st[traj.effector]
        if preempt:
            s['plans'] = []
        if s['plans']:
            last = s['plans'][-1]
            p0, q0 = last.end_pose()
            t0 = max(last.t0 + last.duration, self.t)
        else:
            p0, q0 = s['hold']
            t0 = self.t
        plan = Plan(traj, p0, q0, t0, limits=self.limits, effort_weight=self.effort_weight)
        s['plans'].append(plan)
        s['tripped'] = False
        return plan.pred

    def set_mode(self, mode: str, effector: str = None, **kw) -> None:
        assert mode in ('track', 'impedance', 'guarded', 'force'), mode
        for name in ([effector] if effector else list(self.st)):
            s = self.st[name]
            s['mode'], s['kw'] = mode, dict(kw)
            s['tripped'], s['touched'], s['fi'] = False, False, 0.0
            if mode in ('impedance', 'force'):
                s['ei'][:] = 0

    def set_gripper(self, effector: str, g: float):
        self.st[effector]['grip'] = float(np.clip(g, 0.0, 1.0))

    def command_base(self, target, duration):
        raise NotImplementedError('G1: no walking in v2 (standing balance only)')

    def done(self, effector: str = None) -> bool:
        for n in ([effector] if effector else list(self.st)):
            s = self.st[n]
            if s['tripped']:
                continue
            if s['plans'] and not (len(s['plans']) == 1 and s['plans'][0].done(self.t)):
                return False
        return True

    def tripped(self, effector):
        return self.st[effector]['tripped']

    def time_left(self, effector=None):
        out = 0.0
        for n in ([effector] if effector else list(self.st)):
            for p in self.st[n]['plans']:
                out = max(out, p.t0 + p.duration - self.t)
        return max(out, 0.0)

    def reference(self, effector):
        return self.st[effector]['ref']

    def fallen(self, data=None):
        d = data if data is not None else self.d
        return self.emb.free is not None and float(d.xpos[self.emb.root_body][2]) < self.root_z0 - 0.25

    # ------------------------------------------------------------------ the tick
    def _sensed_ext(self, model, data):
        """generalized forces on the robot from contacts that are not its feet on the floor (actor-legal: wrist F/T
        and tactile skins), passed to the QP as known external forces."""
        emb = self.emb
        ext = np.zeros(model.nv)
        if data.ncon == 0:
            return ext
        g1, g2 = data.contact.geom1, data.contact.geom2
        feet = set(emb.feet_bodies)
        f6 = np.zeros(6)
        own1, own2 = np.isin(g1, emb.geoms), np.isin(g2, emb.geoms)
        tot = np.zeros(3)
        for k in np.where(own1 ^ own2)[0]:
            g = int(g1[k]) if own1[k] else int(g2[k])
            b = int(model.geom_bodyid[g])
            if b in feet:
                continue
            mujoco.mj_contactForce(model, data, int(k), f6)
            fw = data.contact.frame[k].reshape(3, 3).T @ f6[:3]  # on geom2
            if own1[k]:
                fw = -fw
            fw = clip_norm(fw, self.g.get('ext_max', 60.0))
            mujoco.mj_applyFT(model, data, fw, np.zeros(3), data.contact.pos[k], b, ext)
            tot += fw
        # the total sensed wrench handed to the QP is bounded: an over-estimate (penetration spikes) makes the root
        # equations infeasible; the true force is still there and the CoM task absorbs the rest
        n = float(np.linalg.norm(tot))
        if n > self.g['ext_total']:
            ext *= self.g['ext_total'] / n
        return ext

    def step(self, model, data) -> np.ndarray:
        self._ensure(model, data)
        emb, g, wb = self.emb, self.g, self.wb
        dt = self.dt
        self.t = float(data.time)
        qd = data.qvel
        tasks = []
        ext = self._sensed_ext(model, data)
        # ---- balance (floating only): CoM over the feet, root height, root and torso upright (physics_run.py)
        if emb.free is not None:
            com, Jcom = wb.com(), wb.com_jac()
            dJ = np.zeros_like(Jcom) if self.prev_Jcom is None else (Jcom - self.prev_Jcom) / dt
            self.prev_Jcom = Jcom
            ank = np.mean([data.xpos[f.body] + data.xmat[f.body].reshape(3, 3) @ f.centre for f in self.feet], 0)
            # carry the sensed hand forces' moment so the centre of pressure stays inside the soles
            tq = np.zeros(3)
            fz = 0.0
            mid3 = np.array([ank[0], ank[1], 0.0])
            Fext_list = getattr(self, '_fext', [])
            for p_, f_ in Fext_list:
                tq += np.cross(p_ - mid3, f_)
                fz += f_[2]
            want = np.array([-tq[1], tq[0]]) / max(self.weight + fz, 1.0)
            want = np.clip(want, -g['com_shift'], g['com_shift'])
            self.com_off += (want - self.com_off) * min(1.0, dt / 0.05)
            c_ref = ank[:2] + self.com_off
            kp = g['com_kp']
            tasks.append((Jcom[:2], (dJ @ qd)[:2], kp * (c_ref - com[:2]) - 2 * math.sqrt(kp) * (Jcom @ qd)[:2],
                          g['balance_w']))
            self.com_err = float(np.linalg.norm(c_ref - com[:2]))
            jp, jr = wbc_qp.jac(model, data, emb.root_body, data.xpos[emb.root_body])
            dp, dr = wbc_qp.jacdot(model, data, emb.root_body, data.xpos[emb.root_body])
            kz = g['root_z_kp']
            tasks.append((jp[2:], (dp @ qd)[2:], [kz * (self.root_z0 - data.xpos[emb.root_body][2])
                                                  - 2 * math.sqrt(kz) * (jp @ qd)[2]], g.get('root_z_w', g['balance_w'])))
            kr = g['root_kp']
            tasks.append((jr, dr @ qd, kr * wbc_qp.quat_err(self.root_q0, data.xquat[emb.root_body])
                          - 2 * math.sqrt(kr) * (jr @ qd), g['root_w']))
        # torso upright (roll/pitch; its turn is the hands'), both floating and fixed
        _, jt = wbc_qp.jac(model, data, self.torso, data.xpos[self.torso])
        _, dt_ = wbc_qp.jacdot(model, data, self.torso, data.xpos[self.torso])
        zt = data.xmat[self.torso].reshape(3, 3)[:, 2]
        err = np.cross(zt, [0.0, 0.0, 1.0])
        kt = g['torso_kp']
        tasks.append((jt[:2], (dt_ @ qd)[:2], (kt * err - 2 * math.sqrt(kt) * (jt @ qd))[:2], g['torso_w']))
        n_bal = len(tasks)
        # ---- hands
        ee_speed = 0.0
        fext = []
        for name, e in emb.eff.items():
            s = self.st[name]
            plans = s['plans']
            while len(plans) > 1 and plans[0].done(self.t):
                done = plans.pop(0)
                # exact hand-off when queued back-to-back; never earlier than when it was commanded
                plans[0].t0 = max(plans[0].t0, done.t0 + done.duration)
            if plans:
                pl = plans[0]
                if pl.gripper is not None and not getattr(pl, '_grip_applied', False):
                    s['grip'] = float(pl.gripper)
                    pl._grip_applied = True
                tr = pl.traj
                if getattr(tr, 'mode', None) and not getattr(pl, '_mode_applied', False):
                    self.set_mode(tr.mode, name, **tr.mode_kw)
                    pl._mode_applied = True
                pr, vr, ar, qr, wr, alr = pl.at(self.t)
                if pl.done(self.t) and len(plans) == 1:
                    s['hold'] = pl.end_pose()
            else:
                pr, qr = s['hold']
                vr = ar = wr = alr = np.zeros(3)
            p = data.site_xpos[e['site']]
            qm = np.zeros(4)
            mujoco.mju_mat2Quat(qm, data.site_xmat[e['site']])
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, e['site'])
            dp, dr = wbc_qp.jacdot(model, data, e['body'], p)
            v, w = jp @ qd, jr @ qd
            ee_speed = max(ee_speed, float(np.linalg.norm(v)))
            mode, kw = s['mode'], s['kw']
            F = np.zeros(3)
            if mode in ('guarded', 'force') or kw.get('measure_ft') or emb.free is not None:
                F, _, nc = emb.wrist_ft(model, data, name)
                if nc:
                    fext.append((p.copy(), F.copy()))
            s['ft'] = F
            if mode == 'guarded' and not s['tripped']:
                axis = kw.get('axis')
                fmag = float(-(F @ np.asarray(axis))) if axis is not None else float(np.linalg.norm(F))
                if fmag > kw.get('threshold', g['guard_n']):
                    s['tripped'] = True
                    s['trip_t'] = self.t
                    s['hold'] = (p.copy(), qm.copy())
                    s['plans'] = []
            if mode == 'guarded' and s['tripped']:
                pr, qr = s['hold']
                vr = ar = wr = alr = np.zeros(3)
            # contact-limited reference: while the hand is force-capped the reference along the contact normal is
            # pulled back to the hand (no wind-up into a wall); the offset relaxes over ~1 s once contact ends
            co = s.setdefault('cap_off', np.zeros(3))
            Fn0 = float(np.linalg.norm(F))
            fcap0 = kw.get('f_cap', g['f_cap'] + (float(kw.get('force', 0.0)) if mode == 'force' else 0.0))
            if Fn0 > 0.5 * fcap0 and mode != 'force':
                nrm0 = F / Fn0
                depth = float((pr + co - p) @ nrm0)
                if depth < 0:
                    co = co - nrm0 * depth
            elif Fn0 < 1.0:
                co = co * max(0.0, 1.0 - dt / 1.0)
            s['cap_off'] = co
            pr = pr + co
            e_p = clip_norm(pr - p, g['hand_e_max'])
            e_r = quat_err(qr, qm)
            s['err'] = float(np.linalg.norm(pr - p))
            s['ref'] = (pr, vr, qr)
            if mode == 'impedance' or (mode == 'guarded' and s['tripped']):
                K = kw.get('stiffness', g['imp_k'])
                meff = 1.5  # nominal hand effective mass (kg) for an apparent stiffness K
                a_lin = ar + (K * e_p + kw.get('damping', 2 * math.sqrt(K * meff)) * (vr - v)) / meff
                Kr = kw.get('rot_stiffness', g['imp_kr']) / 0.01
                a_rot = alr + Kr * e_r + 2 * math.sqrt(Kr) * (wr - w)
            else:
                s['ei'] = clip_norm(s['ei'] + e_p * dt, g['hand_i_max'])
                kp = g['hand_kp']
                a_lin = ar + kp * e_p + 2 * math.sqrt(kp) * (vr - v) + g['hand_ki'] * s['ei']
                kr = g['hand_kr']
                a_rot = alr + kr * e_r + 2 * math.sqrt(kr) * (wr - w)
            if mode == 'force':
                nax = np.asarray(kw['axis'], float)
                nax = nax / np.linalg.norm(nax)
                Fd = float(kw.get('force', 10.0))
                Fm = float(-(F @ nax))
                vn = float(v @ nax)
                a_lin = a_lin - nax * (a_lin @ nax)
                if not s['touched'] and Fm < kw.get('touch_n', 1.0):
                    a_lin = a_lin + nax * (g['force_kd'] * (kw.get('approach_speed', 0.03) - vn))
                else:
                    s['touched'] = True
                    s['fi'] = float(np.clip(s['fi'] + (Fd - Fm) * dt, -5.0, 5.0))
                    # acceleration in proportion to the force error (physics_run.py press), damped
                    a_lin = a_lin + nax * float(np.clip(0.08 * (Fd - Fm) + g['force_ki'] * 0.02 * s['fi']
                                                        - g['force_kd'] * vn, -6.0, 6.0))
            # contact-force cap (all modes): beyond f_cap the hand yields along the contact normal instead of
            # pushing harder (a balancing body cannot resist an unbounded hand force)
            fcap = kw.get('f_cap', g['f_cap'] + (float(kw.get('force', 0.0)) if mode == 'force' else 0.0))
            Fn_ = float(np.linalg.norm(F))
            s['capped'] = False
            if Fn_ > fcap:
                nrm = F / Fn_            # force on the hand from the environment: points out of the contact
                into = float(a_lin @ nrm)
                if into < 0:
                    a_lin = a_lin - nrm * into
                a_lin = a_lin + nrm * (g['f_back'] * (Fn_ - fcap))
                s['ei'] = s['ei'] - nrm * min(0.0, float(s['ei'] @ nrm))
                s['capped'] = True
            tasks.append((jp, dp @ qd, clip_norm(a_lin, 40.0), g['hand_w']))
            tasks.append((jr, dr @ qd, clip_norm(a_rot, 200.0), g['hand_wr'] * e['cfg'].rot_weight))
        self._fext = fext
        # ---- posture (every QP joint toward its reset value; the arms' home)
        Jpost, apost, wpost = [], [], []
        arm_dofs = self.arm_dofs
        for dof, q0 in self.posture.items():
            if dof in emb.hand_dofs:
                continue
            k = self.k_of_dof[dof]
            J = np.zeros((1, model.nv))
            J[0, dof] = 1
            Jpost.append(J)
            apost.append(g['post_kp'] * (q0 - data.qpos[emb.act_qadr[k]]) - 2 * math.sqrt(g['post_kp']) * qd[dof])
            # fixed base: the legs have no balance job, hold them firmly; waist + arms stay free for the hands
            wpost.append(g['post_w'] if (emb.free is not None or dof in arm_dofs) else 50.0)
        tasks.append((np.vstack(Jpost), np.zeros(len(apost)), np.array(apost), np.array(wpost)))
        tau_qp, f, status = wb.solve(tasks, self.feet, [], ext)
        if status != 'solved':
            # fallback 1: balance + posture only, no sensed forces (the hands go limp for this tick)
            self.qp_status['fallback'] = self.qp_status.get('fallback', 0) + 1
            tau_qp, f, status = wb.solve(tasks[:n_bal] + tasks[-1:], self.feet, [], 0.0 * ext)
            if status != 'solved':
                # fallback 2: also without the joint-range bounds, stance feet as a heavy cost (not an equality)
                self.qp_status['no_ranges'] = self.qp_status.get('no_ranges', 0) + 1
                tau_qp, f, status = wb.solve(tasks[:n_bal] + tasks[-1:], self.feet, [], 0.0 * ext, ranges=False,
                                             soft_feet=True)
            if status != 'solved' and getattr(self, '_tau_qp_ok', None) is not None:
                # fallback 3: keep the last solved torques for this tick
                self.qp_status['hold_prev'] = self.qp_status.get('hold_prev', 0) + 1
                tau_qp = self._tau_qp_ok
        if status == 'solved':
            self._tau_qp_ok = tau_qp
        self.qp_status[status] = self.qp_status.get(status, 0) + 1
        tau_act = np.zeros(len(emb.act_ids))
        for dof, t in zip(wb.adofs, tau_qp):
            k = self.k_of_dof[int(dof)]
            tau_act[k] = t / emb.act_gear[k]
        # ---- fingers: torque-limited joint PD toward open..power-grasp posture, plus their own bias
        for name, e in emb.eff.items():
            s = self.st[name]
            gg = s['grip']
            tgt = e['gopen'] + gg * (e['gclosed'] - e['gopen'])
            if gg >= 0.99:
                tgt = tgt + 0.3 * (e['gclosed'] - e['gopen'])  # overdrive: squeeze when blocked by the object
            qh = data.qpos[e['gqadr']]
            vh = data.qvel[e['gdofs']]
            u = g['finger_kp'] * (tgt - qh) - g['finger_kd'] * vh + data.qfrc_bias[e['gdofs']]
            lim = e['gforce']
            tau_act[e['gacts']] = np.clip(u, -lim, lim) / emb.act_gear[e['gacts']]
        tau_act = np.clip(tau_act, emb.act_lo, emb.act_hi)
        self._meter(model, data, tau_act, ee_speed)
        return tau_act

    def apply(self, model, data):
        u = self.step(model, data)
        data.ctrl[self.emb.act_ids] = u
        return u

    def _meter(self, model, data, tau_act, ee_speed):
        emb, dt = self.emb, self.dt
        span = np.maximum(np.abs(emb.act_lo), np.abs(emb.act_hi))
        tn = tau_act / span
        rate = (tau_act - self.tau_prev) / dt / span if self.tick > 0 else np.zeros_like(tn)
        D = emb.act_dof
        qd = data.qvel[D]
        jerk = (qd - 2 * self.qd1[D] + self.qd2[D]) / (dt * dt) if self.tick >= 2 else np.zeros(len(D))
        av = data.actuator_velocity[emb.act_ids]
        energy = float(np.sum(np.abs(tau_act * emb.act_gear * av)))
        at_lim = bool(np.any(np.abs(tn) >= 0.99))
        self.acc.add(dt, float(tn @ tn), float(rate @ rate), float(jerk @ jerk), energy, at_lim, ee_speed,
                     float(np.max(np.abs(rate))), float(np.max(np.abs(jerk))), float(np.max(np.abs(tn))))
        self.tau_prev = tau_act.copy()
        self.qd2 = self.qd1
        self.qd1 = data.qvel.copy()
        self.tick += 1


# ============================================================================ G1-specific contact skills
# Written against the physmem.skills.SkillContext interface (ctx.emb, ctx.ctrl, ctx.data, ctx.model, ctx.step,
# ctx.wait, ctx.run) so physmem.skills can dispatch to them for the dexterous G1 hands.
# Dex3 pinch geometry (measured, TCP frame): thumb tip and index/middle tips close toward each other along TCP y;
# the pinch centre sits ~(0, +0.025, +0.045) from the TCP (4.5 cm out of the palm, 2.5 cm toward the fingers).
PINCH_CENTRE = np.array([0.0, 0.025, 0.045])


def _axes_quat(z, y):
    z = np.asarray(z, float) / np.linalg.norm(z)
    y = np.asarray(y, float)
    y = y - z * (y @ z)
    y = y / np.linalg.norm(y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([np.cross(y, z), y, z]).reshape(9))
    return q


def _go(ctx, eff, pts, T, q, max_t=None):
    from physmem.embodiments.trajectory import BlendedTrajectory
    ctx.ctrl.command(BlendedTrajectory(effector=eff, waypoints=[(np.asarray(p, float), q) for p in pts], durations=list(T)))
    return ctx.run(max_t=max_t if max_t else sum(T) * 2 + 1.0, effector=eff)


def g1_pinch_pick(ctx, side, est_pos, half, lift=0.10, guard=8.0, squeeze=1.0):
    """Dex3 top-down pinch of a small box (actor-legal estimate `est_pos`, half extents `half`): palm down, thumb
    behind / index+middle in front of the object along +x, guarded descent, force-limited finger PD squeeze, lift.
    -> status ('done' | 'grasp_failed' | 'dropped' | 'fell')."""
    emb, c = ctx.emb, ctx.ctrl
    eff = side
    est = np.asarray(est_pos, float)
    q = _axes_quat((0, 0, -1.0), (1.0, 0, 0))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    centre = np.array([est[0], est[1], est[2] - 0.2 * half[2]])
    tcp = centre - R @ PINCH_CENTRE
    pre = tcp + np.array([0, 0, 0.08])
    c.set_mode('track', eff)
    c.set_gripper(eff, 0.0)
    p0, _ = emb.tcp_pose(ctx.data, eff)
    _go(ctx, eff, [p0 + np.array([0.05, 0, 0.12]), pre], [1.0, 1.2], q)
    c.set_mode('guarded', eff, threshold=guard, axis=(0, 0, -1.0))
    _go(ctx, eff, [tcp], [1.2], q)
    c.set_mode('track', eff)
    c.set_gripper(eff, squeeze)
    ctx.wait(1.0)
    cl, n = emb.gripper_state(ctx.model, ctx.data, eff)
    if n == 0:
        c.set_gripper(eff, 0.0)
        _go(ctx, eff, [pre], [1.0], q)
        return 'grasp_failed'
    here, _ = emb.tcp_pose(ctx.data, eff)
    _go(ctx, eff, [here + np.array([0, 0, lift])], [1.2], q)
    cl, n = emb.gripper_state(ctx.model, ctx.data, eff)
    if c.fallen(ctx.data):
        return 'fell'
    return 'done' if n > 0 else 'dropped'


def g1_tote_carry(ctx, est_pos, half, squeeze_n=30.0, lift=0.10, carry_dy=0.0, hold_s=5.0, mon=None, palm_dx=0.0):
    """two-palm carry of a tote (box, half extents `half`, estimate `est_pos`): palms face each other on the tote's
    +-y sides at mid-height; each palm presses `squeeze_n` along its own normal (hybrid force control: force along
    the squeeze axis, position elsewhere, as CommandAGI physics_run.py's pinch), lift, optional sideways carry,
    hold, put down, release. -> info dict."""
    emb, c, d = ctx.emb, ctx.ctrl, ctx.data
    est = np.asarray(est_pos, float)
    tote_z = est[2]
    g_saved = dict(c.g)
    c.g['root_z_w'] = 8000.0     # keep standing tall: the hands must not pull the pelvis down while loaded
    info = dict(dropped=False, max_lift=0.0)
    q = {}
    ref = {}
    for s, sg in (('left', 1.0), ('right', -1.0)):
        # palm normal (TCP z) points at the tote: left palm faces -y, right palm faces +y; fingers point forward (+x)
        q[s] = _axes_quat((0, -sg, 0), (1.0, 0, 0))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[s])
        R = R.reshape(3, 3)
        # palm centre on the tote side: TCP sits 1 cm out from the side face, hand centred on the tote's x
        side_pt = np.array([est[0], est[1] + sg * (half[1] + 0.01), tote_z])   # palm centre at the tote's COM x
        ref[s] = side_pt - R @ np.array([0.0, palm_dx, 0.0])
    p0 = {s: emb.tcp_pose(d, s)[0] for s in ('left', 'right')}
    from physmem.embodiments.trajectory import BlendedTrajectory
    c.set_mode('track')
    for s, sg in (('left', 1.0), ('right', -1.0)):
        # closed Dex3 = flat palm/fist face (open, the thumb sticks 6 cm out of the palm toward the tote)
        c.set_gripper(s, 1.0)
        # up first (hands rest below table height), then over to the tote's side, then down beside it
        top = tote_z + half[2] + 0.06
        up = np.array([p0[s][0], p0[s][1], top])
        wide = np.array([ref[s][0], ref[s][1] + sg * 0.06, top])
        c.command(BlendedTrajectory(effector=s, waypoints=[(up, q[s]), (wide, q[s]), (ref[s] + np.array([0, sg * 0.06, 0]), q[s])],
                                    durations=[1.2, 1.2, 0.8]))
    ctx.run(max_t=7.0)
    # close in under force control along each palm normal
    for s, sg in (('left', 1.0), ('right', -1.0)):
        c.set_mode('force', s, axis=(0, -sg, 0), force=squeeze_n, approach_speed=0.05, touch_n=2.0)
        c.command(BlendedTrajectory(effector=s, waypoints=[(ref[s], q[s])], durations=[0.8]))
    ctx.run(max_t=2.5)
    ctx.wait(0.8)
    touched = all(c.st[s]['touched'] for s in ('left', 'right'))
    info['touched'] = bool(touched)

    def watch(z0=tote_z):
        if mon:
            mon()

    def lift_both(dz, dy, T):
        for s in ('left', 'right'):
            pr = c.st[s]['ref'][0] if c.st[s]['ref'] is not None else emb.tcp_pose(d, s)[0]
            c.command(BlendedTrajectory(effector=s, waypoints=[(pr + np.array([0, dy, dz]), q[s])], durations=[T]),
                      preempt=True)
        t_end = d.time + T + 0.3
        while d.time < t_end and not c.fallen(d):
            ctx.step()
            watch()
    tote_b = getattr(ctx, 'tote_body', None)
    lift_both(lift, 0.0, 1.5)
    if carry_dy:
        lift_both(0.0, carry_dy, 1.5)
    t_end = d.time + hold_s
    while d.time < t_end and not c.fallen(d):
        ctx.step()
        watch()
    if tote_b is not None:
        info['max_lift'] = float(d.xpos[tote_b][2] - tote_z)
    if carry_dy:
        lift_both(0.0, -carry_dy, 1.5)
    lift_both(-lift + 0.005, 0.0, 1.5)
    for s, sg in (('left', 1.0), ('right', -1.0)):
        c.set_mode('track', s)
        pr = emb.tcp_pose(d, s)[0]
        c.command(BlendedTrajectory(effector=s, waypoints=[(pr + np.array([0, sg * 0.08, 0]), q[s])], durations=[1.0]),
                  preempt=True)
    ctx.run(max_t=2.0)
    c.g.clear()
    c.g.update(g_saved)
    info['fell'] = bool(c.fallen(d))
    return info
