"""v2 shared contracts (lead-owned). Embodiments, skills, environments and models talk only
through these types. Record-layer evidence reuses schema.ObservationEvent / memory.WorldBelief.

Units SI. World frame z-up. Poses as (pos[3], quat_wxyz[4]).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

V2_SCHEMA = "physmem-v2-1"


# ----------------------------------------------------------------- embodiment
@dataclass(frozen=True)
class EmbodimentSpec:
    name: str                       # e.g. "franka_fr3", "ur5e_2f85", "g1_dex3", "aloha_bimanual"
    kind: str                       # "arm" | "bimanual" | "humanoid" | "mobile_manipulator"
    effectors: tuple                # effector names, e.g. ("hand",) or ("left", "right")
    gripper: str                    # "parallel" | "suction" | "dexterous" | "flat_palm"
    reach_m: float
    payload_kg: float
    max_ee_speed: float             # m/s, measured
    max_ee_force: float             # N, measured / spec
    dof: int
    floating_base: bool
    source: str                     # asset provenance (menagerie commit etc.)


@dataclass
class EffortMeter:
    """Accumulated actuator cost over an execution ("servo pain")."""
    torque_sq: float = 0.0          # ∫ Σ (tau_i / tau_max_i)^2 dt
    torque_rate_sq: float = 0.0     # ∫ Σ (dtau_i / tau_max_i)^2 dt
    joint_jerk_sq: float = 0.0      # ∫ Σ (dddq_i)^2 dt  (from encoder finite differences)
    energy_j: float = 0.0           # ∫ Σ |tau_i * dq_i| dt
    time_at_limit_s: float = 0.0    # time any actuator saturated
    peak_ee_speed: float = 0.0

    def cost(self, w=(1.0, 0.1, 1e-4, 0.01, 1.0)) -> float:
        return (w[0] * self.torque_sq + w[1] * self.torque_rate_sq + w[2] * self.joint_jerk_sq
                + w[3] * self.energy_j + w[4] * self.time_at_limit_s)

    def as_dict(self):
        return dict(self.__dict__, cost=self.cost())


class Embodiment(Protocol):
    spec: EmbodimentSpec

    def build(self, mjspec, mount_pose) -> None: ...           # attach robot into a scene MjSpec
    def bind(self, model, data) -> None: ...                   # after compile
    def proprio(self, data) -> dict: ...                       # encoder q/dq, ee poses (FK of encoders), wrist F/T
    def controller(self) -> "Controller": ...


class Controller(Protocol):
    def command(self, traj: "TaskTrajectory") -> None: ...     # queue a task-space trajectory (min-jerk etc.)
    def set_mode(self, mode: str, **kw) -> None: ...           # "track" | "impedance" | "guarded" | "force"
    def step(self, model, data) -> np.ndarray: ...             # -> ctrl (torques), called every control tick
    def done(self) -> bool: ...
    effort: EffortMeter


@dataclass
class TaskTrajectory:
    effector: str
    waypoints: list                 # list of (pos[3], quat[4]) ; first may be None = current
    durations: list                 # seconds per segment
    profile: str = "minjerk"        # velocity profile; segments are blended (no stop at via-points unless stop=True)
    stop_at: tuple = ()             # indices of waypoints where velocity must be zero
    gripper: Optional[float] = None # 0 open .. 1 closed (or suction on/off)


# ----------------------------------------------------------------- skills
@dataclass(frozen=True)
class SkillCall:
    skill: str                      # one of SKILLS_V2 ("reach"|"push"|...|"regrasp"|"move_base")
    params: dict                    # skill-specific, SI, expressed relative to target entity frame where meaningful
    target: Optional[str] = None    # entity id
    effector: Optional[str] = None
    candidate_id: str = ""

    def vector(self, dim: int = 24) -> np.ndarray:
        """Fixed-width numeric encoding shared by all models.

        dim=24 (layout v1, used by existing checkpoints): one-hot over SKILLS_V1 (10) + up to 14 scalar params
        (sorted keys). Raises for skills outside SKILLS_V1.
        dim=32 (layout v2): one-hot over SKILLS_V2 (16 slots, incl. move_base) + up to 16 scalar params."""
        if dim == 32:
            names, n_hot = SKILLS_V2, 16
        else:
            names, n_hot = SKILLS_V1, 10
            if self.skill not in names:
                raise ValueError(f"skill {self.skill!r} needs the v2 encoding: call .vector(dim=32)")
        v = np.zeros(dim, np.float32)
        v[names.index(self.skill)] = 1.0
        vals = [float(self.params[k]) for k in sorted(self.params) if np.isscalar(self.params[k])]
        v[n_hot:n_hot + min(len(vals), dim - n_hot)] = vals[: dim - n_hot]
        return v


SKILLS_V1 = ["reach", "push", "sweep", "pick", "place", "insert", "open", "close", "handover", "regrasp"]
# v2 adds mobile-base motion; remaining slots reserved (append only, never reorder)
SKILLS_V2 = SKILLS_V1 + ["move_base"]


@dataclass
class SkillResult:
    ok: bool                        # primitive executed without controller failure
    status: str                     # "done" | "contact_lost" | "unreachable" | "guard_tripped" | "timeout" | ...
    effort: EffortMeter
    duration_s: float
    log: dict = field(default_factory=dict)


# ----------------------------------------------------------------- environments
@dataclass(frozen=True)
class TaskSpec:
    family: str                     # e.g. "bin_picking", "depalletize", "conveyor_sort", "kitting", "peg_insert", ...
    category: str                   # "bulk" | "fine" | "articulated" | "logistics"
    seed: int
    robot: str
    split: str                      # "train" | "val" | "test" | "heldout_robot" | "heldout_family" | ...
    params: dict = field(default_factory=dict)


@dataclass
class StepOut:
    events: list                    # new ObservationEvents (actor-legal only)
    result: SkillResult
    done: bool
    success: bool
    metrics: dict                   # task progress, collisions, violations
    labels: dict                    # EVALUATOR-ONLY (true masses, frictions, poses, contacts); never fed to the actor


class IndustrialEnv(Protocol):
    task: TaskSpec

    def reset(self, task: TaskSpec) -> list: ...               # -> initial ObservationEvents
    def candidate_skills(self, belief, n: int, rng) -> list: ...  # feasible SkillCalls (no privileged info)
    def execute(self, call: SkillCall) -> StepOut: ...
    def goal(self) -> dict: ...                                 # actor-legal goal description (target regions etc.)
    def render(self, camera: str = "overview") -> np.ndarray: ...
