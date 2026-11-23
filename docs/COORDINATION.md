# sprint coordination (read fully before touching code)

sprint start: 2026-11-23T04:45:53Z (UTC). hard stop: 2026-11-23T16:45Z. all times in ledger are UTC.

## machines
- edit code locally in /home/brandonin/Documents/dynagraph (this repo).
- ALL execution (tests, data generation, training, sim, rendering) runs on host `gb10-direct`
  (promaxgb10-4dfb, GB10 cc 12.1, ~117GB available unified memory, 20 cores) in ~/dynagraph.
- `scripts/sync.sh` rsyncs local -> remote (no delete; never touches remote results/ ckpt/ data/).
- `scripts/remote.sh "<cmd>"` = sync + run in the remote venv with PYTHONPATH=src, MUJOCO_GL=egl.
  e.g. `scripts/remote.sh "python -m pytest -q tests/test_memory.py"`.
  For long jobs: `ssh -f gb10-direct 'cd ~/dynagraph && source .venv/bin/activate && PYTHONPATH=src nohup python ... > logs/X.log 2>&1 &'`
  then poll the log. Never leave an ssh session blocking >2min.
- NEVER `pkill -f <pattern>` over ssh where the pattern appears in your own command line (it kills your ssh). use `pkill -f "[p]attern"`.
- heavy GPU jobs (anything training >1 min on GPU or loading Inkling) MUST go through
  `scripts/gpu_heavy.sh <name> <cmd>` on the remote (flock serialization + ledger logs/gpu_ledger.log).
  CPU MuJoCo work is fine in parallel; keep it to <=8 processes.
- the internet uplink is ~4MB/s shared wifi and a 20GB checkpoint is downloading: do not download anything large.
- remote outputs: ~/dynagraph/results/, ~/dynagraph/data/, ~/dynagraph/logs/. pull back with rsync at the end (lead does it).

## file ownership (exclusive until integration; ask lead via your final report for cross-file changes)
- lead: schema.py, frames.py, memory.py, models/inkling.py, tests/test_memory.py, tests/test_representation.py,
  tests/test_core_integration.py, configs/sprint.yaml, cli.py (shared; others add subcommand functions in their own modules
  and the lead wires them), docs/.
- agent DATA: worlds.py, data.py, tests/test_causality.py, tests/test_data.py.
- agent MODEL: models/compact.py, models/losses.py, models/analytic.py, train.py, evaluate.py, tests/test_models.py.
- agent G1: robots/g1.py, control.py, robots/assets/ (scene xml), tests/test_control_attribution.py, scripts/demo_*.py.
- agent REPORT (later): report/index.html, scripts/make_report.py, results/findings.md.

## science rules (non-negotiable)
- the actor/model never sees simulator-only quantities: true mass, true friction, exact contact impulses,
  future states, perfect contact graphs. Those live only in LabelTensors / evaluator code. Provenance.PRIVILEGED_LABEL
  events must be rejected at actor boundaries (memory.py raises).
- entity/source ids reach models only as random per-sample local indices used via equality. no id embeddings.
- split by root episode id BEFORE windowing/augmentation. scalers fit on train only. record seeds + hashes.
- tune on train/val only. test sets frozen before comparisons. keep every run including failures.
- results rows: results/metrics.jsonl (one json per episode/run outcome, with run_id, model, seed, config_hash,
  status in {observed, projected, target, skipped, untested}).
- label everything honestly: compact model is "compact recurrent physical-memory model", never "inkling".

## the shared task family (decided by lead)
Planar tabletop pushing with hidden per-episode friction (object<->table) and mass. Each ROOT episode =
one object (box, random size), K=4 sequential push primitives, observed at 20Hz (noisy pose sensor for object,
proprioceptive pusher pose, own command), possibly occluded/delayed/dropped. Training SAMPLES = (history up to the
start of push k, action k, future object state at HORIZONS after push k starts). Earlier pushes inform friction (h7:
k=0 has no informative history => uncertainty should be high).
- w1 (identifiable mechanics): puck with known impulses + optional damped spring to an anchor; action kind "impulse".
- w2 (contact + hidden params): box pushed by a primitive spherical pusher (MuJoCo), friction train range 0.3-0.7,
  stress 0.15 and 0.9; mass range e.g. 0.2-2.0kg with held-out combos; pusher geometry variants (sphere r=1.5cm/3cm,
  flat paddle). action kind "push" (theta, phi, speed, duration) in object frame.
- w3 (relational/behavioural, minimal): two boxes connected / a second object in the path, and a moving agent
  with 4 scripted behaviour modes; behaviour latent.
- misspecified law family for e06: velocity-dependent friction (mu(v)=mu0*(1+k*|v|)) and unmodelled actuation drift.
- prediction targets: STATE_DIM=7 (dx, dy, cos dyaw, sin dyaw, vx, vy, wz) in the canonical frame of the target object
  AT THE DECISION TIME (from the actor's latest pose estimate, not the true pose) at HORIZONS (schema.py).
- the G1 endpoint uses the SAME push primitive realized by the G1 hand via IK, and the same selector.

## model factorial (agent MODEL)
b00 generic memory, no physics loss; b10 structured typed memory, no physics; b01 generic + physics; b11 structured + physics.
Plus analytic baseline (Coulomb sliding model with friction estimated from history by least squares) and
constant-velocity/persistence baseline. Oracle (true friction/mass) upper bound labeled separately.
- generic: Perceiver-style latent array (fixed number of latent tokens) updated by cross-attention over raw event tokens
  (same fields), recurrent shared blocks; entity identity only as equality features with the query target.
- structured: per-(entity, property) slots addressed by equality of (entity index, prop) with update gates, explicit
  typed estimate slots for FRICTION and MASS of the target (mean, logvar + latent) that can be READ and OVERWRITTEN
  (interventions), relational set-attention refinement passes k in {1,2,4} (8 = extrapolation), shared weights.
- physics loss: on free-slide horizons (LabelTensors.free_slide_mask): Coulomb deceleration consistency
  |v(t2)| ~= max(0, |v(t1)| - mu_hat*g*(t2-t1)) using the model's OWN friction estimate, plus kinematic consistency
  (displacement ~ trapezoid of predicted velocities), plus friction/mass recovery supervision. Normalized, weights
  fixed from a train/val pilot, logged. Observation (future-state NLL) loss always on for every variant.
- all variants: same data, same heads (common state decoder), param counts within ~10%, same schedule.
