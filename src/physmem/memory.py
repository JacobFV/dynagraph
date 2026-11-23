"""Causal, versioned, auditable property-graph belief store (record layer).

Immutable ObservationEvents are the evidence; PropertyRecords are the current
inferred belief, recomputed from evidence with explicit update semantics:

- causal availability: an event is usable at time `now` only if
  available_at <= now. `ingest_events(..., now)` buffers future-arriving events.
- duplicate suppression: replaying an identical event_id is a no-op.
- privileged labels are rejected at the actor boundary (PrivilegedInputError).
- dynamic properties (POSE2, VEL2, CONTACT, COMMAND): newest *measured* value
  wins; a late-arriving older measurement never overwrites a newer one (it is
  stored as history). Uncertainty grows with age through per-property process
  noise (pose stale -> less reliable).
- static properties (SIZE, MASS, FRICTION, ...): precision-weighted fusion
  ACROSS sources, but repeated evidence from the SAME source is treated as fully
  correlated (only that source's most recent value counts), so neither replays
  nor a chatty sensor manufacture certainty. Static estimates do not decay.
- authoritative corrections replace state immediately, superseding all earlier
  evidence for that property.
- retraction (by event id) and deletion (entity/property) remove the property
  from every readable view; a deleted property is re-inserted only by evidence
  measured after the deletion.
- every change bumps a global revision and the record's revision; records keep
  provenance refs to the events that produced them.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

import numpy as np

from .frames import Frame2, transform_value
from .schema import (
    LATENT_PROPS,
    PROP_DIM,
    Edge,
    EntityDescriptor,
    ObservationEvent,
    Prop,
    PropertyRecord,
    Provenance,
    UpdateKind,
)

DYNAMIC_PROPS = {Prop.POSE2, Prop.VEL2, Prop.CONTACT, Prop.COMMAND, Prop.POSE3, Prop.WRENCH, Prop.JOINT,
                 Prop.GRIPPER, Prop.TACTILE, Prop.EE_POSE}
# process-noise variance growth per second (per component) for dynamic props
PROCESS_NOISE = {
    Prop.POSE2: np.array([0.05**2, 0.05**2, 0.1**2, 0.1**2]),
    Prop.VEL2: np.array([0.2**2, 0.2**2, 0.5**2]),
    Prop.CONTACT: np.array([0.5]),
    Prop.COMMAND: np.array([0.0, 0.0]),
    Prop.POSE3: np.array([0.05**2] * 3 + [0.1**2] * 4),
    Prop.WRENCH: np.array([5.0**2] * 3 + [0.5**2] * 3),
    Prop.JOINT: np.array([0.05**2, 0.5**2]),
    Prop.GRIPPER: np.array([0.01**2, 5.0**2, 0.0]),
    Prop.TACTILE: np.array([0.5, 0.5, 0.5, 5.0**2]),
    Prop.EE_POSE: np.array([0.05**2] * 3 + [0.1**2] * 4),
}


class PrivilegedInputError(ValueError):
    pass


class UnknownPropertyError(KeyError):
    pass


@dataclass
class _Slot:
    events: list = field(default_factory=list)  # usable events (ObservationEvent)
    deleted_at: Optional[float] = None
    authoritative_event: Optional[str] = None
    record: Optional[PropertyRecord] = None
    estimates: list = field(default_factory=list)  # model-written PropertyRecords


class WorldBelief:
    """Per-episode belief. Not thread-safe; one per actor."""

    def __init__(self, episode_id: str, actor_boundary: bool = True):
        self.episode_id = episode_id
        self.actor_boundary = actor_boundary
        self.entities: dict[str, EntityDescriptor] = {}
        self.edges: list[Edge] = []
        self.frames: dict[str, tuple[str, Frame2, str]] = {"world": ("world", Frame2(), "cal-0")}
        self._slots: dict[tuple, _Slot] = defaultdict(_Slot)
        self._seen_ids: set[str] = set()
        self._pending: list[ObservationEvent] = []
        self._retracted: set[str] = set()
        self.revision = 0
        self.now = -math.inf
        self.audit: list[dict] = []

    # ------------------------------------------------------------- structure
    def add_entity(self, desc: EntityDescriptor):
        self.entities[desc.entity_id] = desc
        self._bump("add_entity", desc.entity_id)

    def add_edge(self, edge: Edge):
        if self.actor_boundary and edge.provenance == Provenance.PRIVILEGED_LABEL:
            raise PrivilegedInputError(f"privileged edge {edge} at actor boundary")
        self.edges.append(edge)
        self._bump("add_edge", f"{edge.src}-{edge.rel.name}->{edge.dst}")

    def register_frame(self, frame_id: str, parent: str, T: Frame2, calibration_revision: str = "cal-0"):
        self.frames[frame_id] = (parent, T, calibration_revision)
        self._bump("frame", frame_id)

    def frame_to_world(self, frame_id: str) -> Frame2:
        T = Frame2()
        seen = set()
        fid = frame_id
        chain = []
        while fid != "world":
            if fid in seen or fid not in self.frames:
                raise KeyError(f"unresolvable frame chain from {frame_id}")
            seen.add(fid)
            parent, Tf, _ = self.frames[fid]
            chain.append(Tf)
            fid = parent
        for Tf in reversed(chain):
            T = T.compose(Tf)
        return T

    def transform_between(self, src: str, dst: str) -> Frame2:
        """Pose of frame `src` expressed in frame `dst`."""
        return self.frame_to_world(dst).inverse().compose(self.frame_to_world(src))

    # ------------------------------------------------------------- ingestion
    def _check_event(self, ev: ObservationEvent):
        if self.actor_boundary and ev.provenance == Provenance.PRIVILEGED_LABEL:
            raise PrivilegedInputError(f"privileged event {ev.event_id} ({ev.property_type.name}) at actor boundary")
        if ev.episode_id != self.episode_id:
            raise ValueError(f"event from episode {ev.episode_id} ingested into {self.episode_id}")
        if len(ev.value) != PROP_DIM[ev.property_type]:
            raise ValueError(f"bad value width for {ev.property_type.name}: {len(ev.value)}")
        if ev.units is not None:
            raise ValueError("non-SI units must be converted by the sensor adapter before ingestion")

    def ingest_events(self, events: Iterable[ObservationEvent], now: float, authoritative_sources: Iterable[str] = ()):
        """Assimilate evidence available at `now`. Events whose available_at > now
        are buffered and released by a later call. Returns number of newly applied events."""
        auth = set(authoritative_sources)
        if now < self.now:
            raise ValueError("belief time cannot go backwards")
        self.now = now
        incoming = list(self._pending) + list(events)
        self._pending = []
        applied = 0
        touched = set()
        for ev in incoming:
            self._check_event(ev)
            if ev.available_at > now + 1e-12:
                self._pending.append(ev)
                continue
            if ev.event_id in self._seen_ids or ev.event_id in self._retracted:
                continue  # duplicate packet / replay: not independent evidence
            self._seen_ids.add(ev.event_id)
            key = (ev.entity_id, ev.property_type, ev.related_entity_id)
            slot = self._slots[key]
            if slot.deleted_at is not None and ev.measured_at <= slot.deleted_at:
                continue  # stale evidence cannot resurrect a deleted property
            if ev.source_id in auth:
                slot.authoritative_event = ev.event_id
                # supersede all earlier evidence for this property
                slot.events = [e for e in slot.events if e.measured_at > ev.measured_at]
            elif slot.authoritative_event is not None:
                auth_ev = next((e for e in slot.events if e.event_id == slot.authoritative_event), None)
                if auth_ev is not None and ev.measured_at <= auth_ev.measured_at:
                    continue  # older than the authoritative correction
            slot.events.append(ev)
            touched.add(key)
            applied += 1
        for key in touched:
            self._recompute(key)
        if touched:
            self._bump("ingest", f"{applied} events")
        return applied

    def retract(self, event_id: str):
        self._retracted.add(event_id)
        for key, slot in self._slots.items():
            n = len(slot.events)
            slot.events = [e for e in slot.events if e.event_id != event_id]
            if len(slot.events) != n:
                if slot.authoritative_event == event_id:
                    slot.authoritative_event = None
                self._recompute(key)
                self._bump("retract", event_id)

    def delete(self, entity_id: str, prop: Prop, related_entity_id: Optional[str] = None):
        key = (entity_id, prop, related_entity_id)
        slot = self._slots[key]
        slot.deleted_at = self.now
        slot.events = []
        slot.estimates = []
        slot.authoritative_event = None
        slot.record = None
        self._bump("delete", f"{entity_id}/{prop.name}")

    def write_estimate(self, entity_id: str, prop: Prop, mean, var, source: str, related_entity_id=None, frame_id="world"):
        """Model-written estimate (provenance ESTIMATED). Kept separate from sensor
        evidence; the readable record fuses it only if no stronger evidence exists."""
        key = (entity_id, prop, related_entity_id)
        slot = self._slots[key]
        rec = PropertyRecord(
            entity_id, prop, frame_id, np.asarray(mean, np.float64), np.asarray(var, np.float64), self.now,
            [f"estimate:{source}@{self.now:.3f}"], self.revision + 1, True, UpdateKind.PREDICTED,
            related_entity_id, (source,),
        )
        slot.estimates = [e for e in slot.estimates if e.source_ids != (source,)] + [rec]
        slot.deleted_at = None
        self._recompute(key)
        self._bump("estimate", f"{entity_id}/{prop.name} by {source}")

    # ------------------------------------------------------------- fusion
    def _recompute(self, key):
        slot = self._slots[key]
        entity_id, prop, rel = key
        evs = slot.events
        if not evs and not slot.estimates:
            slot.record = None
            return
        prev_rev = slot.record.revision if slot.record is not None else 0
        if evs and prop in DYNAMIC_PROPS:
            # newest measurement wins; fuse only measurements from distinct
            # sources within one observation interval of the newest
            newest = max(evs, key=lambda e: (e.measured_at, e.received_at))
            group = {}
            for e in evs:
                if newest.measured_at - e.measured_at <= 1e-3:
                    group[e.source_id] = e
            means, precs, refs = [], [], []
            for e in group.values():
                v, var = self._to_world(e)
                means.append(v)
                precs.append(1.0 / np.maximum(var, 1e-12))
                refs.append(e.event_id)
            P = np.sum(precs, 0)
            mean = np.sum([m * p for m, p in zip(means, precs)], 0) / P
            var = 1.0 / P
            rec = PropertyRecord(entity_id, prop, "world", mean, var, newest.measured_at, refs, prev_rev + 1, True,
                                 UpdateKind.OBSERVED if len(group) == 1 else UpdateKind.FUSED, rel,
                                 tuple(sorted(group)))
        elif evs:
            per_source = {}
            for e in sorted(evs, key=lambda e: (e.measured_at, e.received_at)):
                per_source[e.source_id] = e  # same source => correlated: keep latest only
            means, precs, refs = [], [], []
            for e in per_source.values():
                v, var = self._to_world(e)
                means.append(v)
                precs.append(1.0 / np.maximum(var, 1e-12))
                refs.append(e.event_id)
            P = np.sum(precs, 0)
            mean = np.sum([m * p for m, p in zip(means, precs)], 0) / P
            rec = PropertyRecord(entity_id, prop, "world", mean, 1.0 / P,
                                 max(e.measured_at for e in per_source.values()), refs, prev_rev + 1, True,
                                 UpdateKind.OBSERVED if len(per_source) == 1 else UpdateKind.FUSED, rel,
                                 tuple(sorted(per_source)))
        else:
            rec = None
        if slot.estimates:
            est = slot.estimates[-1]
            if rec is None:
                rec = replace(est, revision=prev_rev + 1)
            elif prop in LATENT_PROPS and slot.authoritative_event is None:
                # fuse model estimate (treated as one more independent source)
                P = 1.0 / np.maximum(rec.variance, 1e-12) + 1.0 / np.maximum(est.variance, 1e-12)
                mean = (rec.mean / np.maximum(rec.variance, 1e-12) + est.mean / np.maximum(est.variance, 1e-12)) / P
                rec = replace(rec, mean=mean, variance=1.0 / P, update_kind=UpdateKind.FUSED,
                              provenance_refs=rec.provenance_refs + est.provenance_refs)
        slot.record = rec

    def _to_world(self, e: ObservationEvent):
        v = np.asarray(e.value, np.float64)
        var = np.asarray(e.variance, np.float64)
        if e.frame_id != "world":
            v, var = transform_value(e.property_type, v, var, self.frame_to_world(e.frame_id))
        return v, var

    # ------------------------------------------------------------- queries
    def resolve_query(self, entity_id: str, prop: Prop, target_frame: str = "world", related_entity_id=None,
                      now: Optional[float] = None):
        """Return (mean, var, record) in target_frame at time `now` (defaults to
        belief time). Raises UnknownPropertyError for absent/deleted properties."""
        slot = self._slots.get((entity_id, prop, related_entity_id))
        if slot is None or slot.record is None or not slot.record.valid:
            raise UnknownPropertyError(f"{entity_id}/{prop.name}")
        rec = slot.record
        now = self.now if now is None else now
        var = rec.variance.copy()
        if prop in DYNAMIC_PROPS:
            age = max(0.0, now - rec.estimated_at)
            var = var + PROCESS_NOISE[prop] * age
        mean = rec.mean.copy()
        if target_frame != "world":
            mean, var = transform_value(prop, mean, var, self.transform_between("world", target_frame))
        return mean, var, rec

    def has(self, entity_id: str, prop: Prop, related_entity_id=None) -> bool:
        try:
            self.resolve_query(entity_id, prop, related_entity_id=related_entity_id)
            return True
        except UnknownPropertyError:
            return False

    def available_events(self, now: Optional[float] = None) -> list[ObservationEvent]:
        """All non-retracted, non-deleted evidence available at `now` (for the
        tensorizer). Sorted by measured time."""
        now = self.now if now is None else now
        out = []
        for slot in self._slots.values():
            out.extend(e for e in slot.events if e.available_at <= now + 1e-12)
        return sorted(out, key=lambda e: (e.measured_at, e.event_id))

    def records(self) -> list[PropertyRecord]:
        return [s.record for s in self._slots.values() if s.record is not None and s.record.valid]

    def _bump(self, kind, detail):
        self.revision += 1
        self.audit.append({"rev": self.revision, "t": self.now, "kind": kind, "detail": detail})

    def snapshot(self) -> dict:
        """JSON-friendly view for logs / replay page."""
        return {
            "revision": self.revision,
            "t": self.now,
            "records": [
                {
                    "entity": r.entity_id,
                    "prop": r.property_type.name,
                    "mean": [float(x) for x in r.mean],
                    "var": [float(x) for x in r.variance],
                    "t": r.estimated_at,
                    "rev": r.revision,
                    "kind": r.update_kind.name,
                    "sources": list(r.source_ids),
                    "refs": list(r.provenance_refs)[-3:],
                }
                for r in self.records()
            ],
        }
