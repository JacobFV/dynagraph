"""Whole-body task-space inverse-dynamics QP (ProxQP) for a free-floating (or fixed) robot in MuJoCo.

Copied and adapted from CommandAGI scripts/hardware/wbc.py (same author/org, internal; ProxQP dense solver,
proxsuite BSD-2-Clause). Changes vs the original: a `floating` flag (False = fixed base: no root equations of motion,
no stance contact forces), the QP object is reused between ticks when its size does not change, and the solve
returns the full per-dof torque vector.

Unknowns: joint accelerations (root six first when floating) and a force at each corner of each stance foot's sole.
Constraints: root rows of the equations of motion; every actuated torque within limits; every limited joint within
its range (acceleration bound over 40 ms); corner forces in friction pyramids with a least normal force; no
acceleration of a stance foot (velocity damped to zero). Costs: weighted task accelerations.
"""
import mujoco
import numpy as np
from proxsuite import proxqp
def jac(m, d, body, point):
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, point, body)
    return jp, jr


def jacdot(m, d, body, point):
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacDot(m, d, jp, jr, point, body)
    return jp, jr


def quat_err(qa, qb):
    """The rotation vector (world frame) that takes qb to qa."""
    qi, qe, v = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(qi, qb)
    mujoco.mju_mulQuat(qe, qa, qi)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


class Foot:
    """A foot: its body and its sole's four corners in the body's frame (m)."""

    def __init__(self, m, body):
        self.body = body
        self.corners = np.asarray(sole_corners(m, body), float)
        self.centre = self.corners.mean(0)


def sole_corners(m, body):
    """A foot's sole as four corners in its body's frame: the box, at the lowest height, of the points of its collision
    geoms (a sphere's lowest point, a mesh's vertices, a box's corners) within 4 mm of the lowest."""
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R, p, t = R.reshape(3, 3), m.geom_pos[g], m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(p - np.array([0, 0, m.geom_size[g][0]]))
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            pts += list(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ R.T + p)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[g]
            pts += [p + R @ np.array([a * s[0], b * s[1], -s[2]]) for a in (-1, 1) for b in (-1, 1)]
    if not pts:
        raise ValueError(f'foot {m.body(body).name}: no collision geom to stand on')
    pts = np.asarray(pts)
    low = pts[:, 2].min()
    sole = pts[pts[:, 2] < low + 0.004]
    (x0, y0), (x1, y1) = sole[:, :2].min(0), sole[:, :2].max(0)
    return [[x0, y0, low], [x1, y0, low], [x1, y1, low], [x0, y1, low]]


class WholeBody:
    def __init__(self, m, d, root, dofs, actuated, limits, mu=0.5, fmin=2.0, floating=True):
        """root: the root body; dofs: the dof indices the QP solves for, the root's six first; actuated: the
        torque-driven dofs among them; limits: each actuated dof's (lo, hi) torque (N m)."""
        self.m, self.d, self.root = m, d, root
        self.D = np.asarray(dofs)
        self.n = len(dofs)
        col = {int(v): i for i, v in enumerate(dofs)}
        self.adofs = np.asarray(sorted(actuated, key=lambda v: col[v]))
        self.acol = np.asarray([col[v] for v in self.adofs])
        self.lim = np.asarray([limits[v] for v in self.adofs], float)
        self.mu, self.fmin = mu, fmin
        self.floating = floating
        self.qp, self.qp_shape = None, None
        self.M = np.zeros((m.nv, m.nv))
        self.x = None
        # each limited hinge or slide among the dofs: its column, its qpos address and its range
        self.ranges = []
        for j in range(m.njnt):
            v = int(m.jnt_dofadr[j])
            if v in col and m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.ranges.append((col[v], int(m.jnt_qposadr[j]), *m.jnt_range[j]))

    def com(self):
        return self.d.subtree_com[self.root].copy()

    def com_jac(self):
        jc = np.zeros((3, self.m.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jc, self.root)
        return jc

    def foot_pose(self, f):
        R = self.d.xmat[f.body].reshape(3, 3)
        return self.d.xpos[f.body] + R @ f.centre, R

    def solve(self, tasks, stance, swing, ext, ranges=True, soft_feet=False):
        """One tick. tasks: (J over all nv dofs, J̇·q̇, desired acceleration, weight) each; stance: the feet on the floor;
        swing: (foot, desired 6D acceleration of its sole's centre) for a foot in the air; ext: the sensed generalized
        forces on the robot (all nv) that are not its feet's. Returns ({dof: τ}, the corner forces, the solver's
        status)."""
        m, d, D, n = self.m, self.d, self.D, self.n
        mujoco.mj_fullM(m, d, self.M)
        M = self.M[np.ix_(D, D)]
        h = d.qfrc_bias[D] - d.qfrc_passive[D] - ext[D]
        qd = d.qvel[D]
        Jc = []
        for f in stance:
            R = d.xmat[f.body].reshape(3, 3)
            for c in f.corners:
                Jc.append(jac(m, d, f.body, d.xpos[f.body] + R @ c)[0][:, D])
        nf = 3 * len(Jc)
        Jc = np.vstack(Jc) if Jc else np.zeros((0, n))
        N = n + nf
        P, q = np.zeros((N, N)), np.zeros(N)
        for J, b, a, w in tasks:
            J = J[:, D]
            r = np.asarray(a, float) - np.asarray(b, float)
            w = np.broadcast_to(np.asarray(w, float), r.shape)
            P[:n, :n] += J.T @ (w[:, None] * J)
            q[:n] -= J.T @ (w * r)
        P[:n, :n] += 1e-5 * np.eye(n)
        P[n:, n:] += 1e-6 * np.eye(nf)
        eqA, eqb = [], []
        if self.floating:
            A = np.zeros((6, N)); A[:, :n] = M[:6]; A[:, n:] = -Jc[:, :6].T
            eqA.append(A); eqb.append(-h[:6])
        for f, acc in [(f, None) for f in stance] + list(swing):
            p, _ = self.foot_pose(f)
            jp, jr = jac(m, d, f.body, p)
            dp, dr = jacdot(m, d, f.body, p)
            J6 = np.vstack([jp, jr])[:, D]
            want = -20.0 * (J6 @ qd) if acc is None else np.asarray(acc, float)
            if soft_feet:  # relaxed: stance-foot acceleration as a heavy cost instead of an equality (ROBOTS fix)
                r = want - np.vstack([dp, dr])[:, D] @ qd
                P[:n, :n] += 1e4 * J6.T @ J6
                q[:n] -= 1e4 * J6.T @ r
                continue
            A = np.zeros((6, N)); A[:, :n] = J6
            eqA.append(A); eqb.append(want - np.vstack([dp, dr])[:, D] @ qd)
        C = [np.hstack([M[self.acol], -Jc[:, self.acol].T])]
        lo, hi = [self.lim[:, 0] - h[self.acol]], [self.lim[:, 1] - h[self.acol]]
        mu, big = self.mu, 1e20
        # every joint within its range: an acceleration that, held for 40 ms, stops it short of either end
        if self.ranges and ranges:
            H = 0.04
            A = np.zeros((len(self.ranges), N))
            rl, ru = [], []
            for r, (c, a, jlo, jhi) in enumerate(self.ranges):
                A[r, c] = 1
                qj, vj = d.qpos[a], qd[c]
                a_lo = 2 * (jlo + 0.01 - qj - vj * H) / (H * H)
                a_hi = 2 * (jhi - 0.01 - qj - vj * H) / (H * H)
                if a_lo > a_hi:  # already past a bound at speed: keep the interval non-empty (ROBOTS fix)
                    a_lo = a_hi = 0.5 * (a_lo + a_hi)
                rl.append(a_lo)
                ru.append(a_hi)
            C.append(A); lo.append(rl); hi.append(ru)
        for k in range(nf // 3):
            A = np.zeros((5, N)); i = n + 3 * k
            A[0, i + 2] = 1
            A[1, i], A[1, i + 2] = 1, -mu
            A[2, i], A[2, i + 2] = -1, -mu
            A[3, i + 1], A[3, i + 2] = 1, -mu
            A[4, i + 1], A[4, i + 2] = -1, -mu
            C.append(A); lo.append([self.fmin, -big, -big, -big, -big]); hi.append([big, 0, 0, 0, 0])
        Ae = np.vstack(eqA) if eqA else np.zeros((0, N))
        be = np.concatenate(eqb) if eqb else np.zeros(0)
        C, lo, hi = np.vstack(C), np.concatenate(lo), np.concatenate(hi)
        shape = (N, Ae.shape[0], C.shape[0])
        warm = self.x is not None and len(self.x) == N
        if self.qp is None or self.qp_shape != shape:
            self.qp = proxqp.dense.QP(N, Ae.shape[0], C.shape[0])
            self.qp.settings.eps_abs = 1e-6
            self.qp.settings.max_iter = 2000
            self.qp.settings.verbose = False
            self.qp.settings.initial_guess = proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            self.qp.init(P, q, Ae if len(be) else None, be if len(be) else None, C, lo, hi)
            self.qp_shape = shape
        else:
            self.qp.update(P, q, Ae if len(be) else None, be if len(be) else None, C, lo, hi)
        self.qp.solve()
        qp = self.qp
        status = str(qp.results.info.status).split('.')[-1].removeprefix('PROXQP_').lower()
        x = qp.results.x
        if x is None or not np.all(np.isfinite(x)):
            x = self.x if warm else np.zeros(N)
        self.x = x
        qdd, f = x[:n], x[n:]
        tau = np.clip(M[self.acol] @ qdd + h[self.acol] - Jc[:, self.acol].T @ f, self.lim[:, 0], self.lim[:, 1])
        self.last_qdd = qdd
        return tau, f, status
