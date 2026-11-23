"""Compact episode storage for v2 env data (agent ENVS).

Events are stored per episode as a numpy structured array (+ string tables) instead of pickled ObservationEvent
objects (~5x smaller, fast to load); `events_from_array` rebuilds the exact ObservationEvents (ids, values,
variances, times, sources, provenance), so `memory.WorldBelief` / MODEL-V2's env_adapter can consume them unchanged.

    from physmem.envs.io import load_shard, iter_episodes
    for ep in iter_episodes("data/v2_envs_large/**/train_*.pkl.gz"):   # dicts; ep["events"] -> list[ObservationEvent]
        ...
"""
from __future__ import annotations

import glob
import gzip
import pickle

import numpy as np

from ..schema import ObservationEvent, Prop, Provenance

EV_DTYPE = np.dtype([("ent", "<i4"), ("prop", "<i2"), ("n", "<i1"), ("val", "<f4", (8,)), ("var", "<f4", (8,)),
                     ("t_meas", "<f8"), ("t_recv", "<f8"), ("src", "<i2"), ("prov", "<i1"), ("frame", "<i2"),
                     ("rel", "<i4"), ("eid", "<i8")])


def events_to_array(events):
    ents, srcs, frames = {}, {}, {}
    a = np.zeros(len(events), EV_DTYPE)
    prefix = None
    for i, e in enumerate(events):
        a["ent"][i] = ents.setdefault(e.entity_id, len(ents))
        a["rel"][i] = -1 if e.related_entity_id is None else ents.setdefault(e.related_entity_id, len(ents))
        a["prop"][i] = int(e.property_type)
        n = len(e.value)
        a["n"][i] = n
        a["val"][i, :n] = e.value
        a["var"][i, :n] = e.variance
        a["t_meas"][i] = e.measured_at
        a["t_recv"][i] = e.received_at
        a["src"][i] = srcs.setdefault(e.source_id, len(srcs))
        a["prov"][i] = int(e.provenance)
        a["frame"][i] = frames.setdefault(e.frame_id, len(frames))
        pre, _, num = e.event_id.rpartition(":")
        prefix = pre if prefix is None else prefix
        a["eid"][i] = int(num) if num.isdigit() else i
    return dict(arr=a, ents=list(ents), srcs=list(srcs), frames=list(frames), id_prefix=prefix or "", episode_id=(
        events[0].episode_id if events else ""))


def events_from_array(d):
    a, ents, srcs, frames = d["arr"], d["ents"], d["srcs"], d["frames"]
    out = []
    for r in a:
        n = int(r["n"])
        out.append(ObservationEvent(
            event_id=f"{d['id_prefix']}:{int(r['eid'])}", episode_id=d["episode_id"], entity_id=ents[r["ent"]],
            property_type=Prop(int(r["prop"])), value=tuple(float(x) for x in r["val"][:n]),
            variance=tuple(float(x) for x in r["var"][:n]), frame_id=frames[r["frame"]],
            measured_at=float(r["t_meas"]), received_at=float(r["t_recv"]), source_id=srcs[r["src"]],
            provenance=Provenance(int(r["prov"])), related_entity_id=None if r["rel"] < 0 else ents[r["rel"]]))
    return out


def load_shard(path, as_events=True):
    recs = pickle.load(gzip.open(path))
    if as_events:
        for r in recs:
            if isinstance(r.get("events"), dict):
                r["events"] = events_from_array(r["events"])
    return recs


def iter_episodes(pattern, as_events=True):
    for path in sorted(glob.glob(pattern, recursive=True)):
        for r in load_shard(path, as_events):
            yield r
