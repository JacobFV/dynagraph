"""Attribution / integrity tests for the shared controller and the G1 endpoint (agent G1)."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from physmem import control as C  # noqa: E402
from physmem.robots import g1  # noqa: E402
from physmem.schema import EventTensors, ModelOutput, Prop, Provenance, QueryTensors  # noqa: E402
C.HORIZONS = C.HORIZONS

ALLOWED_ACTOR_PROPS = {Prop.POSE2, Prop.VEL2, Prop.CONTACT, Prop.SIZE, Prop.COMMAND}


def _world(seed=3):
    return g1.G1PushWorld(g1.sample_push_episode(seed), seed=seed)


def _observe(world, t=0.4):
    world.actor.hold(t)
    return world.actor.drain_events()


def _contains_mujoco(obj, depth=0):
    if depth > 3:
        return False
    if type(obj).__module__.startswith("mujoco"):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_mujoco(o, depth + 1) for o in obj)
    if isinstance(obj, dict):
        return any(_contains_mujoco(o, depth + 1) for o in obj.values())
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.Tensor):
        return any(_contains_mujoco(o, depth + 1) for o in vars(obj).values())
    return False


class SpyPredictor:
    """Wraps a predictor; records and checks every argument it is handed."""

    def __init__(self, inner):
        self.inner, self.calls, self.name = inner, 0, "spy_" + inner.name

    def __call__(self, *args, **kw):
        assert not kw, "predictor got keyword args"
        assert len(args) == 2 and isinstance(args[0], EventTensors) and isinstance(args[1], QueryTensors)
        for a in args:
            for k, v in vars(a).items():
                assert isinstance(v, torch.Tensor), f"non-tensor field {k}: {type(v)}"
            assert not _contains_mujoco(a)
        assert not (args[0].provenance == int(Provenance.PRIVILEGED_LABEL)).any()
        self.calls += 1
        return self.inner(*args)


def test_decisions_depend_on_predictor():
    w = g1.G1PushWorld(g1.sample_slide_episode(3), seed=3)  # slide-to-target: friction-sensitive regime
    evs = _observe(w)
    pose = [e for e in evs if e.entity_id == g1.OBJ and e.property_type == Prop.POSE2][-1].value
    half = [e for e in evs if e.entity_id == g1.OBJ and e.property_type == Prop.SIZE][-1].value
    cfg = C.SLIDE_TO_TARGET
    cands = C.generate_candidates(pose, w.goal, w.capability, np.random.default_rng(0), n=48, half=half,
                                  target_entity_id=g1.OBJ, fast_frac=cfg.fast_frac, max_stroke=cfg.max_stroke)
    now = w.port.time
    chosen = {}
    for mu in (0.15, 0.5, 1.2):
        act, info = C.select_action(C.PriorCoulombStub(mu), evs, cands, w.goal, now, g1.OBJ,
                                    rng=np.random.default_rng(1), sd_mode=cfg.sd_mode, push_cost=cfg.push_cost)
        chosen[mu] = (act.candidate.candidate_id, info["order"][:5])
    # same evidence + same candidates, different predictor -> different decision
    assert len({c[0] for c in chosen.values()}) >= 2, chosen
    # same predictor twice -> identical decision (no hidden randomness in selection)
    a1, _ = C.select_action(C.PriorCoulombStub(0.5), evs, cands, w.goal, now, g1.OBJ, rng=np.random.default_rng(1),
                            sd_mode=cfg.sd_mode, push_cost=cfg.push_cost)
    assert a1.candidate.candidate_id == chosen[0.5][0]


def test_episode_integrity_and_predictor_boundary():
    w = _world(0)
    spy = SpyPredictor(C.PriorCoulombStub(0.5))
    res = C.run_episode(w, spy, seed=0)
    a = res.metrics["audit"]
    assert spy.calls == res.metrics["n_pushes"] >= 1
    assert a["external_object_writes"] == 0 and a["eq_active"] == 0 and a["applied_force_steps"] == 0
    assert a["n_equality"] == 0 and a["nan_steps"] == 0
    assert a["object_hash_t0"] != a["object_hash_now"], "box never moved (no real hand-box contact)"
    assert res.metrics["n_contact_events"] > 0
    # the actor/predictor path never held MjData/MjModel
    assert not _contains_mujoco(w.actor.__dict__.get("events"))
    assert "data" not in dir(w.port) and "model" not in dir(w.port)


def test_integrity_monitor_catches_injection():
    w = _world(1)
    sim = w._sim
    sim.step(5)
    sim.data.qpos[sim.box_qadr] += 0.01  # teleport
    sim.step(1)
    assert sim.audit["external_object_writes"] >= 1
    w2 = _world(1)
    s2 = w2._sim
    s2.data.xfrc_applied[s2.box_bid, 0] = 1.0  # force injection
    s2.step(1)
    assert s2.audit["applied_force_steps"] >= 1
    assert not w2.push_success()["clean"]


def test_actor_inputs_not_privileged():
    w = _world(2)
    res = C.run_episode(w, C.PriorCoulombStub(0.5), seed=2, controller=C.ControllerConfig(max_pushes=1))
    w = _world(2)
    w.actor.hold(1.0)
    hist = w.actor.drain_events()
    assert hist
    for e in hist:
        assert e.provenance != Provenance.PRIVILEGED_LABEL
        assert e.property_type in ALLOWED_ACTOR_PROPS, e.property_type  # never MASS / FRICTION
    # ActorPort public surface is minimal and exposes no truth
    public = {n for n in dir(w.port) if not n.startswith("_")}
    assert public == {"time", "episode_id", "encoders", "command", "advance", "drain_events"}
    # decisions log only belief quantities to the actor; truth fields are explicitly eval-only
    for d in res.decisions:
        assert "truth_before_eval_only" in d


def test_capability_speed_range():
    cands = C.generate_candidates((0.4, 0.0, 1.0, 0.0), dict(target_xy=(0.45, 0.05)), g1.G1_CAPABILITY,
                                  np.random.default_rng(0), n=200, half=(0.04, 0.04, 0.02))
    sp = np.array([c.params[2] for c in cands])
    assert sp.max() <= g1.G1_CAPABILITY.max_push_speed + 1e-9 and sp.min() >= C.SPEED_RANGE[0] - 1e-9


def test_slide_to_target_candidates_and_objective():
    cfg = C.SLIDE_TO_TARGET
    assert cfg.max_pushes == 3 and cfg.push_cost > 0 and cfg.sd_mode == "dist"
    ep = g1.sample_slide_episode(4)
    d = math.dist(ep.box.pos, ep.target)
    assert g1.SLIDE_TARGET_DIST[0] - 1e-9 <= d <= g1.SLIDE_TARGET_DIST[1] + 1e-9
    cands = C.generate_candidates((*ep.box.pos, 1.0, 0.0), dict(target_xy=ep.target), g1.G1_CAPABILITY,
                                  np.random.default_rng(0), n=48, half=ep.box.half, fast_frac=cfg.fast_frac)
    fast = [c for c in cands if c.params[2] >= C.FAST_SPEED[0]]
    assert len(fast) >= int(0.4 * 48)
    # 'dist' sd mode ignores variance orthogonal to the goal direction
    mean = torch.zeros(1, len(C.HORIZONS), 7)
    lv = torch.full((1, len(C.HORIZONS), 7), -20.0)
    lv[0, :, 1] = math.log(0.05 ** 2)  # large lateral (y) variance only
    out = ModelOutput(mean=mean, logvar=lv)
    sc_xy = C.score_rollouts(out, (0.0, 0.0, 0.0), dict(target_xy=(0.2, 0.0)), 1.0, "xy")[2][0]
    sc_d = C.score_rollouts(out, (0.0, 0.0, 0.0), dict(target_xy=(0.2, 0.0)), 1.0, "dist")[2][0]
    assert sc_xy > 0.04 and sc_d < 1e-3


def test_inkling_predictor_path_with_tiny_model(tmp_path, monkeypatch):
    """Exercises control.InklingPredictor (load once, chunked bf16 CUDA inference, best-seed-by-val
    selection) with tiny_text_model() + a fake checkpoint. Never uses the real Inkling weights."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    IK = pytest.importorskip("physmem.models.inkling")
    pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [(ord(ch) % 400) + 1 for ch in text[:24]]}

    tm = IK.tiny_text_model("cuda", seed=0)
    monkeypatch.setattr(IK, "load_inkling_text", lambda path=None, **k: (tm, FakeTok(), None, {}))
    monkeypatch.setattr(C, "_INKLING_TEXT", None)
    monkeypatch.setattr(C, "_INKLING_LAST", None)
    prefix = IK.prefix_ids_from_tokenizer(FakeTok()).cuda()
    for seed, nll in ((0, 0.9), (1, 0.4)):
        m = IK.build_inkling_model("i11", tm, prefix, seed=seed)
        sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
        torch.save(dict(trainable=sd, variant="i11", seed=seed, y_mean=torch.zeros(len(C.HORIZONS), 7),
                        y_std=torch.ones(len(C.HORIZONS), 7) * 0.05, val=dict(nll=nll)),
                   tmp_path / f"inkling-main-i11-s{seed}.pt")
    b = C.best_inkling_seed("i11", str(tmp_path))
    assert b[1] == 1  # lower val nll wins
    pred = C.InklingPredictor("i11", "best", ckpt_dir=str(tmp_path))
    assert pred.seed == 1 and "sim paused" in pred.label
    w = g1.G1PushWorld(g1.sample_slide_episode(2), seed=2)
    spy = SpyPredictor(pred)
    res = C.run_episode(w, spy, C.SLIDE_TO_TARGET, seed=2, model_name=pred.name)
    assert spy.calls >= 1 and res.metrics["clean"]
    assert all(d["inference_latency_s"] > 0 for d in res.decisions)
    # loaded once: a second predictor for the same spec reuses the model object
    assert C.InklingPredictor("i11", "best", ckpt_dir=str(tmp_path)).model is pred.model
