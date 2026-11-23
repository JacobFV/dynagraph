"""Shared contracts for the physical-memory sprint.

Everything that crosses a module boundary is defined here. Two layers:

1. Record layer (python dataclasses): immutable ObservationEvent evidence and
   versioned PropertyRecord beliefs, used by memory.py, worlds, the G1 loop.
2. Tensor layer (EventTensors / QueryTensors / LabelTensors): what learned
   models consume. Actor tensors and label tensors are separate types; a model
   forward never receives a LabelTensors object.

Conventions (enforced in frames.py / memory.py, tested in tests/):
- SI units internally (m, kg, s, rad, N). Adapters convert on ingress.
- Planar task frame: z up, gravity -z. Planar pose = (x, y, cos yaw, sin yaw).
- Entity ids and source ids are opaque strings at record level; at tensor level
  they become per-episode local indices assigned in a random order, and models
  may only use them through equality (never as embedding lookups).
- Times at tensor level are ages relative to the query time `now` (seconds,
  >= 0 for available evidence). Absolute time origin never reaches a model.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SCHEMA_REVISION = "physmem-schema-1"

# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------


class Provenance(enum.IntEnum):
    SENSOR = 0
    DERIVED = 1
    ESTIMATED = 2
    PRIVILEGED_LABEL = 3  # never allowed in actor input


class UpdateKind(enum.IntEnum):
    OBSERVED = 0
    FUSED = 1
    PREDICTED = 2
    RETRACTED = 3


class EntityType(enum.IntEnum):
    OBJECT = 0
    PUSHER = 1  # end effector / primitive pusher / robot hand
    FIXTURE = 2  # table, rail, wall, anchor
    AGENT = 3  # other moving agent with latent behaviour
    LINK = 4  # articulated link


class Prop(enum.IntEnum):
    """Property types. `dim` gives the value width in the tensor layer."""

    POSE2 = 0  # x, y, cos(yaw), sin(yaw)       [m, m, -, -]
    VEL2 = 1  # vx, vy, wz                       [m/s, m/s, rad/s] (derived)
    CONTACT = 2  # observed contact indicator p    [-]
    SIZE = 3  # half extents hx, hy, hz          [m]
    COMMAND = 4  # commanded pusher velocity vx, vy [m/s] (actor's own command)
    MASS = 5  # kg   (latent; label-only unless estimated)
    FRICTION = 6  # sliding coefficient, pair-level (object<->support) [-]
    STIFFNESS = 7  # N/m (w1 springs)
    DAMPING = 8  # N s/m
    BEHAVIOR = 9  # categorical probs over scripted agent modes (k=4)
    RELATION = 10  # edge evidence: relation-type one-hot (k=4), see Rel
    # v2 (3-D industrial stations); ids match envs.props.XProp so stored data stays valid
    POSE3 = 100  # x y z qw qx qy qz (world)
    WRENCH = 101  # wrist F/T fx fy fz tx ty tz (world-aligned, tared)
    JOINT = 102  # q, dq of one robot/articulated joint
    GRIPPER = 103  # width, commanded force, suction on
    SHAPE = 104  # one-hot visual category (8)
    TACTILE = 105  # left, right, palm contact flags, |normal force| estimate
    EE_POSE = 106  # effector pose from encoder FK (derived)
    REGION = 107  # cx cy cz hx hy hz yaw (CAD-known fixtures / goal regions)
    GRIP_FRICTION = 120  # latent: object<->gripper friction
    RESTITUTION = 121  # latent
    COMPLIANCE = 122  # latent: contact solref time constant [s]
    CLEARANCE = 123  # latent: fine-task radial clearance [m]
    JOINT_FRICTION = 124  # latent: articulated joint frictionloss


PROP_DIM = {
    Prop.POSE2: 4,
    Prop.VEL2: 3,
    Prop.CONTACT: 1,
    Prop.SIZE: 3,
    Prop.COMMAND: 2,
    Prop.MASS: 1,
    Prop.FRICTION: 1,
    Prop.STIFFNESS: 1,
    Prop.DAMPING: 1,
    Prop.BEHAVIOR: 4,
    Prop.RELATION: 4,
    Prop.POSE3: 7, Prop.WRENCH: 6, Prop.JOINT: 2, Prop.GRIPPER: 3, Prop.SHAPE: 8, Prop.TACTILE: 4,
    Prop.EE_POSE: 7, Prop.REGION: 7, Prop.GRIP_FRICTION: 1, Prop.RESTITUTION: 1, Prop.COMPLIANCE: 1,
    Prop.CLEARANCE: 1, Prop.JOINT_FRICTION: 1,
}
# physical dimension strings (for unit conversion bookkeeping)
PROP_UNITS = {
    Prop.POSE2: ("m", "m", "1", "1"),
    Prop.VEL2: ("m/s", "m/s", "rad/s"),
    Prop.CONTACT: ("1",),
    Prop.SIZE: ("m", "m", "m"),
    Prop.COMMAND: ("m/s", "m/s"),
    Prop.MASS: ("kg",),
    Prop.FRICTION: ("1",),
    Prop.STIFFNESS: ("N/m",),
    Prop.DAMPING: ("N*s/m",),
    Prop.BEHAVIOR: ("1",) * 4,
    Prop.RELATION: ("1",) * 4,
    Prop.POSE3: ("m", "m", "m", "1", "1", "1", "1"), Prop.WRENCH: ("N", "N", "N", "N*m", "N*m", "N*m"),
    Prop.JOINT: ("m|rad", "m/s|rad/s"), Prop.GRIPPER: ("m", "N", "1"), Prop.SHAPE: ("1",) * 8,
    Prop.TACTILE: ("1", "1", "1", "N"), Prop.EE_POSE: ("m", "m", "m", "1", "1", "1", "1"),
    Prop.REGION: ("m", "m", "m", "m", "m", "m", "rad"), Prop.GRIP_FRICTION: ("1",), Prop.RESTITUTION: ("1",),
    Prop.COMPLIANCE: ("s",), Prop.CLEARANCE: ("m",), Prop.JOINT_FRICTION: ("N|N*m",),
}
# latent physical properties an actor may only ESTIMATE (never read from sim)
LATENT_PROPS = {Prop.MASS, Prop.FRICTION, Prop.STIFFNESS, Prop.DAMPING, Prop.BEHAVIOR, Prop.GRIP_FRICTION,
                Prop.RESTITUTION, Prop.COMPLIANCE, Prop.CLEARANCE, Prop.JOINT_FRICTION}
VALUE_DIM = 8  # padded tensor width for any property value
N_PROPS = 11  # v1 dense planar vocabulary (ids 0..10); kept fixed so v1 data/checkpoints stay valid
N_PROPS_V2 = len(Prop)  # all property types incl. v2 3-D ones (sparse ids >= 100); use PROP_INDEX
PROP_INDEX = {p: i for i, p in enumerate(Prop)}  # dense index for embeddings (ids are sparse since v2)


class Rel(enum.IntEnum):
    ATTACHED_TO = 0
    SUPPORTED_BY = 1
    CONTACTING = 2
    JOINT_OF = 3


# --------------------------------------------------------------------------
# record layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationEvent:
    event_id: str
    episode_id: str
    entity_id: str
    property_type: Prop
    value: tuple  # numeric, SI units, in frame `frame_id`
    variance: tuple  # per-component variance (diagonal covariance)
    frame_id: str
    measured_at: float  # s, sensor time
    received_at: float  # s, arrival time at the belief
    source_id: str
    provenance: Provenance = Provenance.SENSOR
    reference_point: str = "com"
    available_from: Optional[float] = None  # defaults to received_at
    validity_interval: float = float("inf")  # s after measured_at
    calibration_revision: str = "cal-0"
    related_entity_id: Optional[str] = None  # for pair/edge properties
    units: Optional[tuple] = None  # None => SI defaults of PROP_UNITS

    @property
    def available_at(self) -> float:
        return self.received_at if self.available_from is None else max(self.available_from, self.received_at)


@dataclass
class PropertyRecord:
    entity_id: str
    property_type: Prop
    frame_id: str
    mean: np.ndarray
    variance: np.ndarray
    estimated_at: float
    provenance_refs: list
    revision: int
    valid: bool = True
    update_kind: UpdateKind = UpdateKind.OBSERVED
    related_entity_id: Optional[str] = None
    source_ids: tuple = ()


@dataclass(frozen=True)
class EntityDescriptor:
    entity_id: str
    entity_type: EntityType


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: Rel
    provenance: Provenance = Provenance.ESTIMATED
    probability: float = 1.0


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Embodiment capability contract (no vendor constants in the core)."""

    name: str
    max_push_speed: float  # m/s
    max_push_duration: float  # s
    workspace_min: tuple  # task-frame xy bounds the effector can reach
    workspace_max: tuple
    effector_radius: float  # m
    action_kinds: tuple = ("push",)


@dataclass(frozen=True)
class ActionCandidate:
    """Shared task-space primitive. Push params are in the OBJECT frame of
    the target entity at the decision time (so they are equivariant by
    construction): contact point on the object boundary given by angle theta
    (rad), push direction relative to the inward normal phi (rad), speed (m/s)
    and duration (s)."""

    candidate_id: str
    kind: str  # "push" | "impulse" (w1 mechanics only) | "noop"
    params: tuple  # push: (theta, phi, speed, duration); impulse: (jx, jy, 0, 0) [N s]
    target_entity_id: str
    feasible: bool = True
    feasibility_note: str = ""

    def vector(self) -> np.ndarray:
        """Canonical 8-d action vector used by every model."""
        v = np.zeros(ACTION_DIM, np.float32)
        if self.kind == "push":
            th, ph, sp, du = self.params
            v[:] = [np.cos(th), np.sin(th), np.cos(ph), np.sin(ph), sp, du, 1.0, 0.0]
        elif self.kind == "impulse":
            jx, jy = self.params[:2]
            v[:] = [0, 0, 0, 0, jx, jy, 0.0, 1.0]
        return v


ACTION_DIM = 8
# prediction horizons (s) relative to action start. primary metric horizons
# are 0.1, 0.5, 1.0
HORIZONS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
PRIMARY_HORIZONS = (0.1, 0.5, 1.0)
# predicted state per horizon, in the canonical (query-object) frame:
# dx, dy, cos dyaw, sin dyaw, vx, vy, wz
STATE_DIM = 7


@dataclass
class PredictedRollout:
    candidate_id: str
    belief_revision: int
    belief_time: float
    horizons: tuple
    mean: np.ndarray  # (H, STATE_DIM), task frame
    var: np.ndarray  # (H, STATE_DIM)
    aux: dict = field(default_factory=dict)  # e.g. {'friction': (m, v)}


@dataclass
class TaskAction:
    candidate: ActionCandidate
    score: float
    rollout: PredictedRollout
    all_scores: dict  # candidate_id -> score


@dataclass
class RobotCommand:
    kind: str
    payload: Any
    source_candidate_id: str


@dataclass
class EpisodeResult:
    episode_id: str
    model: str
    seed: int
    success: bool
    metrics: dict
    decisions: list  # per-decision logs (candidate scores, chosen id, belief rev)


# --------------------------------------------------------------------------
# tensor layer (actor inputs vs labels are distinct types)
# --------------------------------------------------------------------------


@dataclass
class EventTensors:
    """Causally available evidence for B episodes at their query time.

    Shapes: B batch, E max events, N max entities.
      entity   (B,E) long  local entity index (random order per sample)
      related  (B,E) long  related entity index or -1
      prop     (B,E) long  Prop id
      value    (B,E,VALUE_DIM) float  canonical-frame SI values (zero padded)
      logvar   (B,E,VALUE_DIM) float  log variance (padded with 0)
      age      (B,E) float  now - measured_at  (s, >=0)
      latency  (B,E) float  received_at - measured_at (s)
      source   (B,E) long   local source index (equality only)
      provenance (B,E) long
      mask     (B,E) bool   valid & available
      ent_type (B,N) long ; ent_mask (B,N) bool
      dt_obs   (B,) float   nominal observation interval (s)
    """

    entity: Any
    related: Any
    prop: Any
    value: Any
    logvar: Any
    age: Any
    latency: Any
    source: Any
    provenance: Any
    mask: Any
    ent_type: Any
    ent_mask: Any
    dt_obs: Any

    def to(self, device):
        return EventTensors(**{k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.__dict__.items()})


@dataclass
class QueryTensors:
    """target (B,) long entity index whose future is predicted; action (B,A);
    horizons (B,H) seconds."""

    target: Any
    action: Any
    horizons: Any

    def to(self, device):
        return QueryTensors(**{k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.__dict__.items()})


@dataclass
class LabelTensors:
    """Training/evaluator-only. Never passed to a model forward.
      future (B,H,STATE_DIM) canonical-frame future state of target
      future_mask (B,H) bool
      friction (B,) ; mass (B,) ; contact_mode (B,H) long (0 none,1 pushed,2 sliding,3 rest)
      free_slide_mask (B,H) bool  horizons where the declared Coulomb sliding law applies
      law (B,) long  0 coulomb, 1 velocity-dependent (misspecified), ..."""

    future: Any
    future_mask: Any
    friction: Any
    mass: Any
    contact_mode: Any
    free_slide_mask: Any
    law: Any

    def to(self, device):
        return LabelTensors(**{k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.__dict__.items()})


@dataclass
class ModelOutput:
    """Every model (compact b00..b11, analytic, inkling i00/i11) returns this.
      mean, logvar (B,H,STATE_DIM)
      friction_mean, friction_logvar, mass_mean, mass_logvar (B,) or None
      contact_logits (B,H,4) or None
      slots: optional dict exposing typed memory for interventions/inspection
    """

    mean: Any
    logvar: Any
    friction_mean: Any = None
    friction_logvar: Any = None
    mass_mean: Any = None
    mass_logvar: Any = None
    contact_logits: Any = None
    slots: Optional[dict] = None
    cost: Optional[dict] = None
