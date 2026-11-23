"""v2 property vocabulary. The lead APPROVED the extension: schema.Prop now contains POSE3=100 ... JOINT_FRICTION=124
(same ids / dims as the former envs XProp table), memory.WorldBelief ingests them. This module is kept as a thin
compatibility alias: `XProp is schema.Prop`; every emitted event is schema-standard."""
from __future__ import annotations

from ..schema import PROP_DIM, Prop

XProp = Prop
SHAPES = ("box", "cylinder", "capsule", "sphere", "ellipsoid", "mesh", "carton", "tote")
XPROP_DIM = {p: PROP_DIM[p] for p in Prop if int(p) >= 100}
XLATENT = {Prop.GRIP_FRICTION, Prop.RESTITUTION, Prop.COMPLIANCE, Prop.CLEARANCE, Prop.JOINT_FRICTION}


def prop_dim(p) -> int:
    return PROP_DIM[Prop(int(p))]


def is_standard(ev) -> bool:
    return int(ev.property_type) in {int(p) for p in Prop}


def split_events(events):
    """(standard, unknown); with the approved schema every env event is standard."""
    std, ext = [], []
    for e in events:
        (std if is_standard(e) else ext).append(e)
    return std, ext


def ingest(belief, events, now):
    std, ext = split_events(events)
    belief.ingest_events(std, now)
    return ext
