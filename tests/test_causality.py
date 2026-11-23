"""Causality / leakage tests for the data pipeline (agent DATA)."""
import copy
import json
import os

import numpy as np
import pytest
import torch

from physmem import data as D
from physmem import worlds as W
from physmem.schema import Provenance, ObservationEvent, Prop


def _root(split="train", idx=3, seed=7):
    root = D.root_int_for(split, idx)
    ss = np.random.SeedSequence([seed, root])
    r_world, r_sensor, r_obs = [np.random.default_rng(s) for s in ss.spawn(3)]
    ep = W.sample_episode(r_world, D.STRATA[split]["world"])
    traj = W.simulate(ep)
    scfg = D._sample_sensor(r_sensor, D.STRATA[split]["sensor"])
    return ep, traj, scfg, root, ss


def _eq(a, b):
    for k in ("ent", "rel", "prop", "val", "logvar", "age", "latency", "src", "prov", "mask", "est"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("split", ["train", "occ_1.0", "delay_3", "w3", "w1"])
def test_future_poison_does_not_change_prefix_tensors(split):
    ep, traj, scfg, root, ss = _root(split)
    rng_obs = lambda: np.random.default_rng(ss.spawn(3)[2])
    cols = D.observe(ep, traj, scfg, np.random.default_rng(np.random.SeedSequence([7, root]).spawn(3)[2]), root)
    for k in range(W.K_PUSHES):
        now = float(traj["push_start"][k]) - D.DECISION_EPS
        ref = D.tensorize_sample(cols, now)
        # (1) poison every not-yet-available event by a huge value
        pc = copy.deepcopy(cols)
        fut = pc["avail"] > now
        pc["val"][fut] = 1e9
        pc["var"][fut] = 1e9
        pc["meas"][fut] = now - 1.0  # even claiming an old measurement time must not leak
        _eq(ref, D.tensorize_sample(pc, now))
        # (2) poison the simulator's future (ground truth after now) and re-run the sensor model
        pt = copy.deepcopy(traj)
        tf = pt["t"] > now - D.DECISION_EPS + 1e-9
        for key in ("obj_pose", "obj_vel", "pusher_pose", "pusher_vel", "f_push", "cmd"):
            pt[key][tf] = 1e6
        cols2 = D.observe(ep, pt, scfg, np.random.default_rng(np.random.SeedSequence([7, root]).spawn(3)[2]), root)
        _eq(ref, D.tensorize_sample(cols2, now))
        # (3) future labels are poisoned but actor tensors only depend on events
        lab = D.make_labels_np(pt, ep, k, ref["est"])
        assert np.abs(lab["future"]).max() > 1e5  # labels did see the poison, tensors did not


def test_record_path_future_poison():
    ep, traj, scfg, root, ss = _root("train", 5)
    cols = D.observe(ep, traj, scfg, np.random.default_rng(1), root)
    evs = D.columnar_to_events(cols, root, D.entity_types_for(ep))
    now = float(traj["push_start"][2]) - D.DECISION_EPS
    poisoned = [e if e.available_at <= now else ObservationEvent(**{**e.__dict__, "value": tuple(1e9 for _ in e.value)})
                for e in evs]
    a = D.events_to_tensors([evs], [now], np.random.default_rng(0))
    b = D.events_to_tensors([poisoned], [now], np.random.default_rng(0))
    for k, v in a.__dict__.items():
        assert torch.equal(v, b.__dict__[k]), k


def test_privileged_event_raises():
    ep, traj, scfg, root, ss = _root("train", 1)
    cols = D.observe(ep, traj, scfg, np.random.default_rng(1), root)
    evs = D.columnar_to_events(cols, root, D.entity_types_for(ep))
    priv = ObservationEvent(event_id="leak", episode_id=evs[0].episode_id, entity_id=evs[0].entity_id,
                            property_type=Prop.FRICTION, value=(ep.friction,), variance=(0.0,), frame_id="world",
                            measured_at=0.0, received_at=0.0, source_id="sim", provenance=Provenance.PRIVILEGED_LABEL)
    with pytest.raises(D.PrivilegedEventError):
        D.events_to_tensors([evs + [priv]], [1.0], np.random.default_rng(0))
    # even if it would only become available in the future
    late = ObservationEvent(**{**priv.__dict__, "received_at": 100.0, "measured_at": 100.0})
    with pytest.raises(D.PrivilegedEventError):
        D.events_to_tensors([evs + [late]], [1.0], np.random.default_rng(0))
    pc = copy.deepcopy(cols)
    pc["prov"][0] = int(Provenance.PRIVILEGED_LABEL)
    with pytest.raises(D.PrivilegedEventError):
        D.tensorize_sample(pc, 1.0)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("data"))
    D.generate(dict(name="tiny", n_train=16, n_val=8, n_test=8, n_strata=4, procs=4, out_root=root,
                    strata=["ood_fric_0.9", "w1", "w3"]))
    return root


def _all_roots(d):
    out = {}
    for f in os.listdir(d):
        if f.endswith("_tensors.pt"):
            out[f[: -len("_tensors.pt")]] = set(torch.load(os.path.join(d, f), weights_only=True)["root"].tolist())
    return out


def test_splits_share_no_root(tiny_dataset):
    d = os.path.join(tiny_dataset, "tiny")
    roots = _all_roots(d)
    names = sorted(roots)
    assert {"train", "val", "test"} <= set(names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (roots[a] & roots[b]), (a, b)
        # event store and label store agree on roots of the split
        ev = set(np.load(os.path.join(d, f"{a}_events.npz"))["root"].tolist())
        lb = set(np.load(os.path.join(d, f"{a}_labels.npz"))["root"].tolist())
        assert ev == lb == roots[a]


def test_full_dataset_splits_disjoint_if_present():
    d = os.path.join(D.DATA_DIR, "v1")
    if not os.path.exists(os.path.join(d, "manifest.json")):
        pytest.skip("v1 not generated")
    roots = _all_roots(d)
    names = sorted(roots)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (roots[a] & roots[b]), (a, b)


def test_scales_from_train_only(tiny_dataset):
    d = os.path.join(tiny_dataset, "tiny")
    with open(os.path.join(d, "scales.json")) as f:
        saved = json.load(f)
    assert saved["fit_split"] == "train"
    tr = torch.load(os.path.join(d, "train_tensors.pt"), weights_only=True)
    va = torch.load(os.path.join(d, "val_tensors.pt"), weights_only=True)
    assert json.loads(json.dumps(D.fit_scales(tr))) == saved
    assert json.loads(json.dumps(D.fit_scales(va))) != saved
    assert saved["n_samples"] == len(tr["action"])
