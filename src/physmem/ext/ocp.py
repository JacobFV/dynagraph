"""Physion Object Contact Prediction (OCP) readout: reproduction of the official protocol (owner: agent EXT).

Official sources (both MIT):
  * neuroailab/physics-models physion/objective/objective.py: ReadoutObjective.get_readout_model
        Pipeline([StandardScaler(), LogisticRegression(max_iter=MAX_ITER)]),
        GridSearchCV over clf__C = np.logspace(*LOGSPACE), cv = StratifiedKFold(CV, shuffle=True, random_state=seed)
  * neuroailab/physopt-physics-benchmarking physopt/config.py defaults:
        LOGSPACE = (-8, 8, 17), CV = 5, MAX_ITER = 200, NORM_INPUT = True, PROTOCOLS = observed / simulated / input
  * configs/*/physion.yaml: SEEDS [0, 1, 2]; readouts fit per scenario (paper §2.4 and README: "Readouts are always
    trained in a per-scenario fashion").
Protocol names (paper §2.4): "observed" = features of the first 45 frames only; "observed+simulated" = observed
features concatenated with the model's predicted future features; "full" = features of the entire movie (a detection
task, not prediction; not comparable with humans).

Metrics (paper §2.4; our implementation of the stated definitions): accuracy of the binary prediction; Pearson r
between P(contact) and the mean human response per stimulus; Cohen's kappa between the model's binary predictions and
each participant's responses, averaged over participants.
"""
from __future__ import annotations

import numpy as np

LOGSPACE = (-8, 8, 17)
CV = 5
MAX_ITER = 200
SEEDS = (0, 1, 2)
PROTOCOLS = ("observed", "observed+simulated", "full")


def readout_model(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=MAX_ITER))])
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, param_grid={"clf__C": np.logspace(*LOGSPACE)}, cv=skf)


def cohen_kappa(a, b) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def metrics(pred, proba, y, human: dict | None = None, names=None) -> dict:
    """pred (N,) bool, proba (N,) P(contact), y (N,) bool; human: stim -> array of 0/1 responses (one per
    participant; we pair responses by stimulus, so kappa is per-stimulus agreement averaged -- see docs/EXT.md)."""
    pred = np.asarray(pred, bool)
    y = np.asarray(y, bool)
    out = dict(n=int(len(y)), accuracy=float(np.mean(pred == y)))
    if human and names is not None:
        hm = np.array([np.mean(human[n]) if n in human else np.nan for n in names])
        ok = np.isfinite(hm)
        if ok.sum() > 2:
            out["pearson_r_human_mean"] = float(np.corrcoef(np.asarray(proba)[ok], hm[ok])[0, 1])
            out["n_human_stimuli"] = int(ok.sum())
    return out


def kappa_vs_participants(pred, names, participant_table) -> float:
    """participant_table: dict participant -> dict stim -> 0/1. Mean over participants of Cohen's kappa between the
    model's binary predictions and that participant's responses on the stimuli both saw (paper's definition)."""
    ks = []
    pm = dict(zip(names, np.asarray(pred, bool)))
    for _, resp in participant_table.items():
        common = [n for n in resp if n in pm]
        if len(common) >= 10:
            ks.append(cohen_kappa([pm[n] for n in common], [resp[n] for n in common]))
    return float(np.mean(ks)) if ks else float("nan")


def fit_eval(X_tr, y_tr, X_te, y_te, seeds=SEEDS, human=None, names_te=None, participants=None) -> dict:
    """official readout over seeds; returns per-seed metrics and the seed mean."""
    runs = []
    for s in seeds:
        m = readout_model(s).fit(X_tr, np.asarray(y_tr, int))
        pred = m.predict(X_te).astype(bool)
        proba = m.predict_proba(X_te)[:, 1]
        r = metrics(pred, proba, y_te, human, names_te)
        if participants is not None and names_te is not None:
            r["cohen_kappa_mean"] = kappa_vs_participants(pred, names_te, participants)
        r["best_C"] = float(m.best_params_["clf__C"])
        r["seed"] = s
        runs.append(r)
    keys = [k for k in runs[0] if isinstance(runs[0][k], float) and k not in ("best_C",)]
    return dict(runs=runs, mean={k: float(np.nanmean([r[k] for r in runs])) for k in keys})


def bootstrap_ci(correct: np.ndarray, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    b = rng.choice(c, (n_boot, len(c))).mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
