"""EXT unit tests: synthetic data only (no datasets needed)."""
import numpy as np
import pytest

from physmem.ext import ocp
from physmem.ext.tracks import KIND_OBJECT, TrackSet, resample, to_v3_evidence


def _scene(T=60, H=128, W=128, contact=True):
    """red square moving right toward a static yellow square."""
    rgb = np.full((T, H, W, 3), (40, 60, 110), np.uint8)
    rgb[:, :, :, :] = (40, 60, 110)
    for t in range(T):
        x = 10 + t * (1.2 if contact else 0.3)
        rgb[t, 60:70, int(x):int(x) + 10] = (200, 10, 25)
    rgb[:, 58:72, 90:104] = (235, 215, 70)
    return rgb


def test_color_tracker_and_features(tmp_path):
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from physmem.ext import frontend_color as FC
    ts = FC.track_frames(_scene())
    assert ts.valid[:, 0, 0].all() and ts.valid[:, 1, 0].all()
    assert abs(ts.uv[0, 0, 0, 0] - 14.5) < 1.0 and abs(ts.uv[0, 1, 0, 0] - 96.5) < 1.0
    o, s = FC.ocp_track_features(ts, 45)
    assert np.isfinite(o).all() and np.isfinite(s).all()
    p = tmp_path / "a.npz"
    ts.save(p)
    ts2 = TrackSet.load(p)
    assert np.allclose(np.nan_to_num(ts.uv), np.nan_to_num(ts2.uv)) and ts2.entities == ["agent", "patient"]


def test_resample_and_v3_evidence():
    T = 30
    t = np.arange(T) / 30.0
    uv = np.zeros((T, 1, 1, 2), np.float32)
    uv[:, 0, 0, 0] = 2.0 * t
    uv[10:12] = np.nan                     # short dropout is interpolated
    ts = TrackSet(KIND_OBJECT, t, uv, ["a"], ["c"], (10, 10), "synthetic")
    g = np.linspace(0, t[-1], 20)
    X = resample(ts, g)
    assert np.allclose(X[:, 0, 0, 0], 2.0 * g, atol=1e-5)
    ev = to_v3_evidence(ts, g)
    assert ev.shape == (1, 20, 2, 6)
    assert np.allclose(ev[0, 2:-2, 0, 1], 2.0, atol=1e-3) and ev[..., 5].all()


def test_official_readout_shapes():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = X[:, 0] + 0.3 * rng.normal(size=120) > 0
    r = ocp.fit_eval(X[:80], y[:80], X[80:], y[80:], seeds=(0,))
    assert r["mean"]["accuracy"] > 0.8
    assert ocp.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_h6_lift_detection_synthetic():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "h6f", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ext" / "h6_features.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T, fps = 75, 30.0
    t = np.arange(T)
    # camera-frame: table plane y = 0.5 (camera looks along +z, y down); up = -y
    h = np.where(t < 30, 0.20 - 0.004 * t, np.where(t < 40, 0.08, 0.08 + 0.006 * (t - 40)))
    xyz = np.zeros((T, 21, 3))
    xyz[:, :, 1] = 0.5 - h[:, None]
    xyz[:, :, 2] = 0.8
    xyz[:, 8, 0] = 0.05                                 # index tip 5 cm from thumb tip
    n, p0 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.5, 0.8])
    f = H.features(xyz, fps, n, p0)
    assert f["lift_detected"] and 40 <= f["t_lift"] <= 46 and 25 <= f["t_grasp"] <= f["t_lift"]
    assert abs(f["c1_aperture"] - 0.05) < 1e-6 and f["c2_peak_vz"] > 0.15
    # plane fit on a synthetic depth image of a tilted table
    K = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])
    v, u = np.mgrid[0:96, 0:128]
    z = 0.8 + 0.002 * (v - 48)
    nn, pp, inl = H.table_normal(z, K, np.random.default_rng(0))
    assert inl > 0.9 and nn @ (-pp) > 0
