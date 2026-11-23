"""Kinematic IK (damped least squares) on a scratch MjData: reachability checks and home postures. Never touches the
live simulation state."""
from __future__ import annotations

import mujoco
import numpy as np

from physmem.embodiments.trajectory import quat_err


def solve_ik(emb, model, data, eff, pos, quat=None, q0=None, iters=200, tol=1e-3, rot_tol=0.02, rot_weight=None,
             scratch=None):
    """-> (q, ok, pos_err, rot_err). Joint limits respected by clamping."""
    e = emb.eff[eff]
    d = scratch if scratch is not None else mujoco.MjData(model)
    d.qpos[:] = data.qpos
    if q0 is not None:
        d.qpos[e['qadr']] = q0
    D, A = e['dofs'], e['qadr']
    rw = e['cfg'].rot_weight if rot_weight is None else rot_weight
    lo = np.array([model.jnt_range[model.dof_jntid[k]][0] if model.jnt_limited[model.dof_jntid[k]] else -1e9 for k in D])
    hi = np.array([model.jnt_range[model.dof_jntid[k]][1] if model.jnt_limited[model.dof_jntid[k]] else 1e9 for k in D])
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    ep = er = np.inf
    for it in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        p = d.site_xpos[e['site']]
        qm = np.zeros(4)
        mujoco.mju_mat2Quat(qm, d.site_xmat[e['site']])
        dp = np.asarray(pos) - p
        dr = quat_err(quat, qm) if quat is not None else np.zeros(3)
        ep, er = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        if ep < tol and (quat is None or er < rot_tol):
            break
        mujoco.mj_jacSite(model, d, jp, jr, e['site'])
        if quat is not None:
            J = np.vstack([jp[:, D], rw * jr[:, D]])
            r = np.concatenate([dp, rw * dr])
        else:
            J, r = jp[:, D], dp
        lam = 1e-3 + 0.05 * min(1.0, ep)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), r)
        dq = np.clip(dq, -0.3, 0.3)
        d.qpos[A] = np.clip(d.qpos[A] + dq, lo, hi)
    return d.qpos[A].copy(), bool(ep < 5 * tol and (quat is None or er < 5 * rot_tol)), ep, er


def down_quat(yaw=0.0, closing_yaw=None):
    """TCP pointing straight down (z = -world z), closing axis (TCP y) at angle `yaw` in the horizontal plane."""
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    x = np.cross(y, z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).reshape(9))
    return q
