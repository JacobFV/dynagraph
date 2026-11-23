"""Units and frame transformations for typed property values.

Planar task frame (z up). A frame is an SE(2) transform (x, y, yaw) plus a
length scale relative to SI metres. Transform rules are declared per property
type so that vectors, poses, twists and scalars transform consistently:

  POSE2    : position rotates+translates, yaw adds
  VEL2     : linear part rotates (free vector; galilean shift NOT applied here,
             see `galilean_shift`), angular rate unchanged
  COMMAND  : free vector, rotates
  SIZE     : body-frame half extents, invariant under frame change (scales)
  CONTACT, MASS, FRICTION, BEHAVIOR, RELATION : invariant scalars
  STIFFNESS: N/m  -> scales with 1/length ; DAMPING N s/m -> 1/length

Variances transform with the same Jacobian (diagonal approximation for
rotations: the xy block is rotated as a full 2x2 covariance and re-diagonalised
only if the caller requests a diagonal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema import Prop

LENGTH_UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
MASS_UNITS = {"kg": 1.0, "g": 0.001}


@dataclass(frozen=True)
class Frame2:
    """Pose of frame F expressed in the world frame W: p_W = R(yaw) p_F + t."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def rot(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def inverse(self) -> "Frame2":
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Frame2(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.yaw)

    def compose(self, other: "Frame2") -> "Frame2":
        """self ∘ other: other expressed in self, returns other in self's parent."""
        p = self.rot() @ np.array([other.x, other.y]) + np.array([self.x, self.y])
        return Frame2(float(p[0]), float(p[1]), self.yaw + other.yaw)


def _rot_cov(R: np.ndarray, var2: np.ndarray) -> np.ndarray:
    C = R @ np.diag(var2) @ R.T
    return np.diag(C).copy()


def transform_value(prop: Prop, value, variance, T: Frame2):
    """Map a value expressed in frame A to frame B, where T = pose of A in B.

    Returns (value, variance) as float64 arrays.
    """
    v = np.asarray(value, np.float64).copy()
    var = np.asarray(variance, np.float64).copy()
    R = T.rot()
    if prop == Prop.POSE2:
        p = R @ v[:2] + np.array([T.x, T.y])
        c, s = math.cos(T.yaw), math.sin(T.yaw)
        cy, sy = v[2], v[3]
        v = np.array([p[0], p[1], c * cy - s * sy, s * cy + c * sy])
        var[:2] = _rot_cov(R, var[:2])
        # (cos,sin) components: rotate the 2-vector the same way
        var[2:4] = _rot_cov(R, var[2:4])
    elif prop in (Prop.VEL2,):
        v[:2] = R @ v[:2]
        var[:2] = _rot_cov(R, var[:2])
    elif prop == Prop.COMMAND:
        v[:2] = R @ v[:2]
        var[:2] = _rot_cov(R, var[:2])
    # SIZE, CONTACT, MASS, FRICTION, ... : invariant under rigid frame change
    return v, var


def convert_length_units(prop: Prop, value, variance, from_scale: float, to_scale: float = 1.0):
    """Rescale a property's length-bearing components. `from_scale` = metres per
    input unit (e.g. 0.001 for mm)."""
    k = from_scale / to_scale
    v = np.asarray(value, np.float64).copy()
    var = np.asarray(variance, np.float64).copy()
    if prop == Prop.POSE2:
        v[:2] *= k
        var[:2] *= k * k
    elif prop in (Prop.VEL2,):
        v[:2] *= k
        var[:2] *= k * k
    elif prop in (Prop.SIZE,):
        v *= k
        var *= k * k
    elif prop == Prop.COMMAND:
        v[:2] *= k
        var[:2] *= k * k
    elif prop == Prop.STIFFNESS:  # N/m = kg s^-2 ; per length unit
        v /= k
        var /= k * k
    elif prop == Prop.DAMPING:
        v /= k
        var /= k * k
    return v, var


def galilean_shift(prop: Prop, value, variance, u, t: float):
    """Inertial-observer change: new frame moves with constant velocity u (m/s)
    relative to the old one; at time t positions shift by -u t and velocities by -u.
    Valid only for the declared Newtonian model (moving boundaries must be shifted
    too by the caller)."""
    v = np.asarray(value, np.float64).copy()
    u = np.asarray(u, np.float64)
    if prop == Prop.POSE2:
        v[:2] -= u * t
    elif prop == Prop.VEL2:
        v[:2] -= u
    return v, np.asarray(variance, np.float64).copy()


def quat_canonical(q):
    """Canonicalise a unit quaternion (w,x,y,z) sign: q and -q are the same
    orientation; returns the representative with w >= 0."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quat_to_yaw(q) -> float:
    w, x, y, z = quat_canonical(q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_to_cs(yaw: float):
    return math.cos(yaw), math.sin(yaw)


def cs_to_yaw(c: float, s: float) -> float:
    return math.atan2(s, c)


def pose_to_frame(pose_value) -> Frame2:
    """POSE2 value (x,y,c,s) -> Frame2 of that body in its parent frame."""
    x, y, c, s = pose_value[:4]
    return Frame2(float(x), float(y), math.atan2(float(s), float(c)))
