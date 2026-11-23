# DATA: worlds, observation streams, tensors, splits (agent DATA)

> **v3 = EXPLORATORY / POST-HOC** (`~/dynagraph/data/v3`, generated 2026-11-23 07:48Z, 55.9 s on 8 procs). Motivated by
> the v2 result that friction was never recovered or used by the models. Identical to v2 except push speed
> log-U[0.4, 1.0] m/s and a NEW base seed 20261125 (so v3 roots differ from v2's). Only these splits/strata exist:
> train (4096 roots), val, test, ood_fric_0.15, ood_fric_0.9, misspec_velfric, occ_0.5, paddle (512 each).
> Any result on v3 must be reported as exploratory and post hoc (the dataset was designed after seeing v2 results).
> See the "v3" section at the end for hashes and measurements.

> **v2 is the PRIMARY dataset** (`~/dynagraph/data/v2`, `load_split(name, dataset="v2")`). v1 is kept untouched for
> reference only. Reason: in v1 (push speed U[0.05,0.4] m/s, pusher holding at push end) friction was barely
> identifiable — free slides were 0.1-1.5 cm and <0.15 s, so friction 0.35->0.65 moved the final pose only 1-8 mm.
> v2 changes ONLY the push regime so that friction changes the consequence of an action:
> speed log-U[0.05,1.0] m/s, duration U[0.1,0.5] s, the pusher RETRACTS at push end (0.2 m/s backwards for 0.1 s,
> visible as COMMAND events) so the box slides freely, pusher force limit 100 N (was 30), box half-height
> hz in [1.5 cm, 0.9*min(hx,hy)] so fast-decelerating boxes do not tip. Everything else (ranges, strata, seeding,
> splits, scale fitting on train) is identical; same push timing (t_k = 0.3 + 2.3k).
>
> v2 measurements (from the stored label store): pushes making contact 97% (train); box at rest by h=2.0 s in
> 100.00% of samples of every stratum; tipped 0% (0.5% in misspec_velfric). Post-release slide (push end -> t_k+2 s,
> contact pushes, train) percentiles p25/50/75/90/99/max = 0.0/0.2/0.8/2.3/8.1/18.9 cm (mu=0.15: 0.2/0.7/2.5/7.9/23.8/42.2;
> mu=0.9: 0.0/0.2/0.5/1.2/4.7/8.2; paddle 0.1/0.5/2.3/5.6/11.5/16.2). Samples with >=1 free-slide horizon: train 35.7%
> (per horizon 0/0.13/0.33/0.36/0.36/0.36), val 35.1, test 35.1, ood_fric_0.15 79.0, ood_fric_0.9 24.2, heldout_cells
> 38.4, paddle 54.0, misspec_drift 35.4, w3 33.6, misspec_velfric 0 (by definition). Because speed is log-uniform the
> median slide is still small (most pushes are slow: a 0.2 m/s release at mu=0.5 stops within 4 mm / 0.04 s); long
> slides come from the fast tail. Coulomb check (test): 0.8 m/s push, release v~0.63-0.74 m/s: mu=0.3 slide 8.7-9.5 cm
> vs v^2/2mu g 8.6-9.3 cm (ratio 1.02); mu=0.7 3.2 cm vs 2.9-3.0 cm (ratio 1.10-1.13).
> v2 generation: 106.3 s on 8 procs (train 24.4 s). Hashes below (v2 table) and in v2/manifest.json.

Code: `src/physmem/worlds.py` (simulators), `src/physmem/data.py` (sensors, tensorizer, labels, generation, loaders).
Primary dataset: `gb10-direct:~/dynagraph/data/v1/` (3.1 GB, generated 2026-11-23 ~05:15Z in 98.6 s on 8 procs).

## Loading (training)

```python
from physmem import data as D
tr = D.load_split("train", dataset="v1", device="cuda")      # PhysSplit, whole split resident on GPU
for ev, q, lab in D.make_loader(tr, batch_size=256, shuffle=True, seed=0):  # EventTensors, QueryTensors, LabelTensors
    ...
va = D.load_split("val", "v1", device="cuda"); D.make_loader(va, 512, shuffle=False, permute=True, seed=1)
scales = D.load_scales("v1")      # train-only normalization stats (dict, see below)
manifest = D.load_manifest("v1")
```
- `make_loader(split, batch_size=256, shuffle=True, seed=0, device=None, permute=True, drop_last=False, dataset="v1")`
  re-shuffles and re-draws the random per-sample entity / source local-index permutations every epoch
  (generator seeded by `(seed, epoch)`). A full GPU epoch over train (16384 samples, bs 256) takes ~0.1 s; CPU ~0.5 s.
- `PhysSplit.batch(idx, generator, permute)` gives one batch for arbitrary indices; `PhysSplit.meta(idx)` -> root id, push index k.
- Record path (memory.py / G1 loop): `events_to_tensors(list_of_event_lists, now_list, rng, max_events=512, permute=True,
  targets=None, entity_types=None, dt_obs=None, return_info=False)` and `make_query(target_index, actions, horizons)`.
  `root_events(split, i, dataset)` returns the stored record-layer `ObservationEvent` list of a root.
  `make_labels_np(traj, ep, k, est)` + `labels_to_tensors([...])` build labels from a ground-truth trajectory.

## Samples
Root episode = one world instance, K=4 push primitives starting at t_k = 0.3 + 2.3 k s (episode 9.5 s).
Sample (root, k): history = all events with `available_at <= now`, `now = t_k - 1 ms`; query = action k
(`ActionCandidate.vector()`), horizons `schema.HORIZONS` after t_k; label = true target state at t_k+h.

Canonical frame = actor's latest available POSE2 **estimate** of the target (by measured_at) at `now` — never the
true pose. Labels `future[h] = (dx, dy, cos dyaw, sin dyaw, vx, vy, wz)` of the TRUE target relative to that estimate.
Event values (POSE2 position/heading, VEL2, COMMAND vectors and their variances) are expressed in the same frame.

### Tensorizer (`tensorize_sample`, shared by record path and cache)
1. raise `PrivilegedEventError` if any `Provenance.PRIVILEGED_LABEL` event is present (available or not);
2. causal filter `available_at <= now`; 3. dedupe by `event_id` (earliest-available copy kept);
4. canonical frame transform; 5. cap at `max_events=512`: the newest event of every (entity, prop, source) group is
always kept (so the one-shot SIZE events survive), the remaining budget is filled with the most recent events;
deterministic (sort by measured_at, event_id); 6. `age = now - measured_at`, `latency = received_at - measured_at`,
`logvar = log(max(var,1e-12))` (0 beyond the prop's width); 7. random per-sample permutation of entity indices
(N_MAX=4) and source indices (S_MAX=12) applied at batch time; `QueryTensors.target` is the permuted target index.
Padding: entity=0, source=0, related=-1, mask=False. `ent_type` padded with 0 and `ent_mask` False.
Cap is hit in 5.5% of train samples (all at k=3), 50% at 40 Hz.

## Worlds
**w2** box (half extents U[2,5] cm each, pos U[-0.1,0.1]^2 m, yaw U[-pi,pi]) on a plane, pushed by a pusher at z=2 cm.
Hidden: sliding friction mu (train U[0.3,0.7]), mass (train log-U[0.2,2.0] kg). Held-out (mass bin, friction bin)
cells on a 4x4 grid (log-mass edges 0.2,0.36,0.63,1.12,2.0; friction edges 0.3,0.4,0.5,0.6,0.7): (3,0) heavy+slippery and
(0,3) light+grippy are excluded from train/val/test and form `heldout_cells`.
Push primitive `(theta, phi, speed, duration)`: theta U[-pi,pi] (ray from box centre to boundary point), phi U[-0.4,0.4]
(rotation of the inward face normal), speed U[0.05,0.4] m/s, duration U[0.1,0.6] s (5 ms grid). The pusher is
re-positioned 0.25 s before t_k to boundary point + outward normal*(pusher half-thickness + 3 mm), using the TRUE box
pose (well-calibrated executor), then moves at `speed` along the push direction for `duration`, then holds.
Pushers: sphere r=1.5 cm, sphere r=3 cm (train), flat paddle 8x70x30 mm oriented face-on (held out, `paddle`).

MuJoCo (4.0.1) contact model: timestep 2.5 ms, `cone="elliptic"`, `impratio=1`, `integrator="implicitfast"`,
condim=3 (tangential Coulomb only), default solref (0.02, 1) / solimp (0.9, 0.95, 0.001). Friction mixing by geom
`priority`: box (priority 1, friction mu) determines box<->table; pusher (priority 2, friction 0.3) determines
pusher<->box. Box<->plane gives up to 4 corner contacts, so frictional torque emerges. Pusher = slide x/y + hinge
joints, velocity actuators kv=300, |F|<=30 N, outer position loop `ctrl = v_cmd + 20 (x_ref - x)` (strongly
position-controlled; it lags slightly under heavy loads, so mass is only weakly observable — in quasi-static pushing
and Coulomb free sliding the kinematics are nearly mass-independent; expect mass recovery to be poor, this is
physics, not a bug). Pusher collides with boxes only. Sanity (tests): box at rest drifts < 0.2 mm over 9.5 s;
free slide after a 0.3 m/s push: mu=0.3 -> 14.7 mm (Coulomb v^2/2mu g = 15.2), mu=0.7 -> 6.3 mm (6.5).
Re-simulating a globally SE(2)-transformed world changes canonical labels by <= 1.4 mm.

**Misspecified laws (e06)**: `misspec_velfric` law=1: box geom friction overwritten every step to
mu0(1+k|v_xy|), k~U[1,3] s/m (MuJoCo re-reads geom friction at collision time each step).
`misspec_drift` law=2: realized pusher reference velocity = commanded + constant per-episode bias
(|b| U[0.02,0.05] m/s, random world direction) during pushes; COMMAND events report the commanded value only.
**w1** (negative control, numpy): point-mass puck (radius U[2,4] cm) with Coulomb friction mu U[0.03,0.15], mass
U[0.5,2] kg, known impulses (action kind "impulse", |J| U[0.03,0.3] N s, object frame) at t_k; 50% of roots have a
damped spring (k U[2,20] N/m, c U[0.1,1] N s/m) to an anchor (FIXTURE entity, observed once, plus a RELATION
ATTACHED_TO event) -> law=3. No pusher entity.
**w3** (minimal): w2 + a second box (own pose sensor, OBJECT entity index 2 canonical) placed 0.5-5 cm beyond the
target along the first push direction (contact rewiring). No behavioural agent (not done).

## Observation streams (record layer, all synthetic)
| entity | prop | source | details |
|---|---|---|---|
| target | POSE2 | pose sensor | rate r, sigma 2 mm / 1 deg (var on cos,sin = sigma_yaw^2), occlusion, drop, delay, dup |
| target | VEL2 | derived (prov DERIVED) | causal finite difference of consecutive RECEIVED pose events (dt<=0.3 s), var 2 sigma^2/dt^2 |
| target | SIZE | size estimator | once, measured 0, received 0.05 s, sigma 2 mm |
| pusher | POSE2 | proprioception | rate r, sigma 0.5 mm / 0.5 deg, latency 0.2-1 ms, never dropped |
| pusher | SIZE | pusher config | once (exact geometry half extents) |
| pusher | COMMAND | own command | at each change (push start/end), world-frame commanded velocity, latency 0 |
| pusher | CONTACT | synthetic tactile | rate r/2, 1 if (true pusher-box normal force + N(0,0.3 N)) > 0.5 N, 2% flips, var 0.05 |
| obj2 (w3) | POSE2/VEL2/SIZE | own sensors | as target |
| anchor (w1) | POSE2 | map | once |
Corruptions on the object pose sensor: occlusion (one window of length occ per push period, never before t=0.3),
drop prob, delay = d observation intervals (+0.5-3 ms jitter), duplicates (5%: same event_id replayed 0.3-2 intervals
later). Train/val/test mix per root: occ {0,0,0.2,0.5}, delay {0,0,1}, drop {0,0,0.25}, rate 20 Hz.

## Strata files (512 roots = 2048 samples each)
`ood_fric_0.15`, `ood_fric_0.9`, `heldout_cells`, `paddle`, `rate_10|20|40` (clean sensors), `occ_0.0|0.2|0.5|1.0`,
`delay_0|1|3`, `drop_0.0|0.25|0.5` (one factor at a time, others clean, 20 Hz), `misspec_velfric`, `misspec_drift`,
`w1`, `w3`. Root ids = split_code*100000 + idx (disjoint by construction; checked in tests); seeds
`SeedSequence([base_seed=20261123, root_id])` split into (world, sensor-config, sensor-noise) streams.

## Labels (LabelTensors; evaluator/training only)
`future (B,6,7)`, `future_mask` (all True), `friction`, `mass`, `law` (0 coulomb, 1 velocity-dependent friction,
2 actuation drift, 3 w1 spring), `contact_mode (B,6)`: 1 pusher normal force > 0.05 N, 0 tipped (>0.2 rad),
2 moving (|v|>5 mm/s or |wz|>0.05 rad/s), 3 rest. `free_slide_mask (B,6)`: law in {0,2}, t_k+h >= push_end+0.05 s,
target speed at push_end+0.05 s > 5 mm/s, and no pusher/other-object contact and no tilt > 0.1 rad on
[push_end+0.05, t_k+h] (includes horizons after the box came to rest by Coulomb deceleration). Coverage: 40% of train
samples have >=1 free-slide horizon (0.108/0.324/0.403 at h=0.25/0.5/>=1.0); 83% for mu=0.15; 0% for misspec_velfric.
Note: boxes are at rest by h=1.0 in essentially all w2 samples (velocity std 0 at h>=1).

## Files per split (`~/dynagraph/data/v1/`)
- `{split}_events.npz` actor stream: columnar raw events (eid, ent, rel, prop, val[8], var[8], meas, recv, avail, src,
  prov) concatenated over roots with `offsets`, `root`, `ent_types`, `dt_obs`.
- `{split}_labels.npz` label store: ground truth at 20 Hz (`traj_*`: obj pose/vel/tilt, pusher pose, contact forces),
  push_start/end. `{split}_params.jsonl`: hidden episode params + sensor config (simulator-only).
- `{split}_tensors.pt` pre-tensorized canonical (unpermuted) samples + labels used by the loader.
- `scales.json` (fit on TRAIN only: per-prop value mean/std, age, latency, action, future mean/std per horizon,
  friction, log-mass), `manifest.json` (config, strata definitions, seeds, root ranges, sha256 of every file, timings).

## v1 hashes (sha256 prefix: tensors / events / labels), generation seconds
| split | roots | samples | sec | tensors.pt | events.npz | labels.npz |
|---|---|---|---|---|---|---|
| train | 4096 | 16384 | 23.17 | 818ce2c6b840e32c | a5cc395c7acbc798 | 41c86b9929a4e25e |
| val | 512 | 2048 | 2.9 | 7b906d9b791a9a2e | 8ad0b12e4bb32359 | 958d32a40ced64ff |
| test | 512 | 2048 | 2.9 | 0867248bcce6d8b7 | a0edaeaf6dcf2438 | c6f701926e4fef16 |
| ood_fric_0.15 | 512 | 2048 | 2.94 | 08717400e57861cb | 3ed88d5a68ea25e8 | 2238fad94fdeac93 |
| ood_fric_0.9 | 512 | 2048 | 2.94 | 99995b6b4ccc243e | 4368b1220ae17c77 | 6ae21841e01e44ad |
| heldout_cells | 512 | 2048 | 2.92 | dbedacf57ef8a0bf | fcb7d25896d2a06b | 838bcdcb511eccf3 |
| paddle | 512 | 2048 | 3.16 | 218c66bfbe16bb32 | a49738c91fca91a4 | 6d3ffe91c948820a |
| rate_10 | 512 | 2048 | 2.93 | e66c0c3728029e9a | 3570a737a509bf2e | ffe5c1dedfca95e5 |
| rate_20 | 512 | 2048 | 3.12 | 12b7cc2cbec917c3 | 82f235f18112b7a4 | 9eef63b98e92c339 |
| rate_40 | 512 | 2048 | 3.14 | 0d69474547b509f4 | 700f82e44f5a9368 | 329530c2f9bed7c2 |
| occ_0.0 | 512 | 2048 | 3.2 | 0fb14477627957a1 | 95fc820f1ffb50d7 | c05139c0655eba9f |
| occ_0.2 | 512 | 2048 | 3.1 | 5f9b98333b5d3ad9 | 89cb8f843ff82444 | 548f69766997d5ad |
| occ_0.5 | 512 | 2048 | 2.99 | b0c2ae9ed2917cc1 | 47940247b7411a21 | 611d8f3dd473a5b2 |
| occ_1.0 | 512 | 2048 | 2.97 | af0026f8f2117c49 | 43af0ddf545cc772 | aaf29d8d87ade841 |
| delay_0 | 512 | 2048 | 2.97 | cba98b5a36653498 | 31734f4295de28e4 | 2a9cb16363e939df |
| delay_1 | 512 | 2048 | 3.03 | d6fa85bc9c1f8912 | 9f62bf037f1b9a62 | cf47468051acda06 |
| delay_3 | 512 | 2048 | 3.36 | d6265e017fe508a0 | 14ea44918df46d58 | 2db7ab7c702f60a3 |
| drop_0.0 | 512 | 2048 | 3.12 | 5bb25570115560e0 | 9d99ea4103489f7b | d846803de9b82181 |
| drop_0.25 | 512 | 2048 | 3.12 | 876f568dc6acfdd5 | 07bc8a49f8758e4c | 8f0b976ca9fb90b4 |
| drop_0.5 | 512 | 2048 | 3.13 | 1e9fd70865c36669 | c673fc4e1ecae6e6 | 49a26edd423671c5 |
| misspec_velfric | 512 | 2048 | 3.11 | 1fc617f6d9eb8fd9 | 582e8c1a926c0dc9 | 8b568df8440cfc40 |
| misspec_drift | 512 | 2048 | 3.15 | b301e2c28fbb2238 | 84fda18f09b9c5cb | cffa6d389c8c0225 |
| w1 | 512 | 2048 | 0.88 | 45eb69ecbca52954 | 2c4c9fa1f6d596b4 | d48d70d4af647bf7 |
| w3 | 512 | 2048 | 4.68 | ca7b4e678d367506 | f6b9b687fc777eac | 5058ea2c70264073 |
scales.json sha256 prefix 7ff703e65f7ceac1. Full hashes in manifest.json.

Regenerate: `python -m physmem.data --name v1 --procs 8` (flags `--n_train --n_val --n_test --n_strata --strata a,b|none`)
or `physmem.data.generate(dict(name=..., ...))`.

## v2 hashes (sha256 prefix: tensors / events / labels), generation seconds (PRIMARY)
| split | roots | samples | sec | tensors.pt | events.npz | labels.npz |
|---|---|---|---|---|---|---|
| train | 4096 | 16384 | 24.42 | 050c2e6a7b4443da | 453e61fd9bb0cb27 | 7befad39d65c2cb9 |
| val | 512 | 2048 | 3.39 | b112c18e0b5b873c | 489105cefd609fd6 | b4addd9b5f5170c6 |
| test | 512 | 2048 | 3.2 | 99dcd6383edba1b2 | 5f0f7b8b195cb282 | 94ac98186e3e9328 |
| ood_fric_0.15 | 512 | 2048 | 3.38 | 2933c76551aeca39 | 1a26f28d3788a180 | 3b09a467cf5e6261 |
| ood_fric_0.9 | 512 | 2048 | 3.07 | f0f2456084318395 | 72973586bd3d416f | e57b7c34510d407b |
| heldout_cells | 512 | 2048 | 3.11 | db0087f8bbe30a91 | 00dba5aa37a59f23 | faeb962eb82cac4a |
| paddle | 512 | 2048 | 3.35 | b454b595f802068d | e702ee9d8771863b | 9d738144e47f319d |
| rate_10 | 512 | 2048 | 3.3 | 38ab8179c7ebe06f | 47d8b0e4a91e151a | abc39dd779d5997b |
| rate_20 | 512 | 2048 | 3.25 | e48d64cf883752ef | 1c6ed798b4937c07 | fe1745bfb9df3d82 |
| rate_40 | 512 | 2048 | 3.17 | 3f6304a0b8d493b1 | 80a99a8a4bfff12e | e5a2660c3f499cba |
| occ_0.0 | 512 | 2048 | 3.4 | 37a0eb8d27d0e171 | 62b369606bba2cf5 | b6e872fe14bba62d |
| occ_0.2 | 512 | 2048 | 3.13 | 24f09f6b569e5f91 | 5eb404fa1836bca6 | 1a26e358724895a6 |
| occ_0.5 | 512 | 2048 | 3.28 | f811894263feaff7 | 27dfba1aad3f0a1c | e978e16290e81f07 |
| occ_1.0 | 512 | 2048 | 3.19 | 9f6dc7ab297f35cf | f0f081f1143447e1 | a3d1d1b45a6cf494 |
| delay_0 | 512 | 2048 | 3.38 | 2f500fb6a0217e39 | 38601130c2a5b9ae | 7c3654beda763f43 |
| delay_1 | 512 | 2048 | 3.08 | d4216c6634bf7f29 | f83195df72ea8d12 | 764c27750c89fe58 |
| delay_3 | 512 | 2048 | 3.26 | 360898ae465907f4 | 0f4a751174e67fbc | 26a85fe92c03d4c4 |
| drop_0.0 | 512 | 2048 | 3.21 | 69fa6948c52d1ea8 | 50096421518918ac | f56ff2328ec4ccf7 |
| drop_0.25 | 512 | 2048 | 3.08 | 33bd629aa3a83da4 | 63820f9eead02158 | 86352ccbdd11f780 |
| drop_0.5 | 512 | 2048 | 3.07 | 09ae718ab52ed284 | e9457614a5bd5cd8 | fd757176d6c0ac50 |
| misspec_velfric | 512 | 2048 | 3.26 | 5622a038e43216e3 | 707bcee06982cfa9 | a3a9db2e5894cc7b |
| misspec_drift | 512 | 2048 | 3.2 | 0ce2d894c4ef00b0 | 0e48a20358fab28a | baf0311dbad0fe17 |
| w1 | 512 | 2048 | 0.96 | 45eb69ecbca52954 | 2c4c9fa1f6d596b4 | d48d70d4af647bf7 |
| w3 | 512 | 2048 | 4.54 | 42b98a414f8d58b9 | caa8241a4e76ad23 | 18c8b6443e4a5185 |
v2 scales.json sha256 prefix a8a9d5b01e14aa66. (w1 is identical to v1: the w1 world is unchanged.)
Regenerate: `python -m physmem.data --name v2 --procs 8`.

## v2 follow-up strata (held-out TEST only; train data unchanged)
**chain_L1 / chain_L2 / chain_L4 / chain_L8 (e10)** — `world="chain"`: a line of L boxes along the chain axis
(= target yaw), gaps U[0,4] mm, lateral jitter +-4 mm, yaw jitter +-0.05 rad; intermediate boxes have half extents
hx U[2,4] cm, hy U[2.5,5] cm, stable hz, mass log-U[0.2,1.0] kg, friction U[0.3,0.7] (hidden, not labelled).
The query TARGET is the LAST box (canonical entity 0, hidden mass/friction from the usual v2 ranges); the push is
applied to the FIRST box, aimed through the chain at the target: at re-position time (0.25 s before t_k, i.e. before
`now`) theta = direction from first box away from the target (in the first box's frame) and phi rotates the push onto
the first-box->target line (|phi|<=0.6). The query action vector uses these REALIZED (theta, phi) and the v2
speed/duration; theta/phi are thus in the frame of the pushed (first) box, whose yaw ~= the target's at k=0.
For L=1 the target itself is pushed along the chain axis. Entities: 0 target, 1 pusher, 2.. chain boxes (first pushed =
2); L=8 => 9 entities (ent_type width = max(4, L+1) per file; the loader uses the file's width). Extra boxes are seen
by the same camera model (source SRC_OBJ2_POSE, derived VEL2). Labels: `chain_L` in labels.npz / tensors.pt (meta).
Note: with many entities the 512-event cap binds earlier (keeps ~1.5 s of history at k=3 for L=8).

**agent (e16 / h7)** — v2 w2 + a kinematic disc agent (mocap, r=2.5 cm, AGENT entity index 2) with a LATENT
behaviour mode sampled uniformly per root: 0 stationary (12-20 cm from target), 1 crossing (ping-pong along a line
passing within 5 cm of the target, +-25 cm, 0.04-0.1 m/s), 2 approach (moves toward the current target position at
0.03-0.06 m/s, stops at r + max(hx,hy) + 5 mm), 3 oscillating (sinusoid, amp 4-8 cm, period 1.5-3 s, centre
12-18 cm away). The agent collides with boxes (not the pusher/table); contact counts as non-table contact (breaks
free_slide). All agents are static until t=T0=0.3 s, so the k=0 history carries no information about the mode.
Observed like an object (pose sensor + derived VEL2 + SIZE). Mode id is stored ONLY in the label store
(`agent_mode` in labels.npz, tensors.pt, params.jsonl; exposed via `PhysSplit.meta()`), never as an event
(no BEHAVIOR events are emitted). Views: `load_split("agent_k0", "v2")` (k=0 samples) and
`load_split("agent_klate", "v2")` (k>=1).

**Nested subsets (e17)** — `v2/subsets.json`: train ROOT ids for 1pct/10pct/100pct = nested prefixes of a
seed-1234 permutation of the 4096 train roots (41 / 410 / 4096 roots), with sample indices and sha256 of each root list.
Load with `load_split("train", "v2", subset="10pct")`; or `make_subsets(dataset, seed)` to rebuild.

### follow-up strata: hashes, timings, stats (added to v2 at 05:26Z; 47.9 s on 8 procs)
| split | roots | samples | sec | tensors.pt | events.npz | labels.npz | cap hit | ent width | target moved >5 mm by 2 s | median disp | free-slide any |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_L1 | 512 | 2048 | 3.4 | eb006552424f789b | 669346d1f977c2bf | d7f027bb6e290e83 | 8% | 4 | 99% | 6.6 cm | 53% |
| chain_L2 | 512 | 2048 | 5.1 | 8eeb1376c73974bb | 07c4259729c76030 | f10229bdc369e0c3 | 36% | 4 | 95% | 6.2 cm | 46% |
| chain_L4 | 512 | 2048 | 9.9 | 2126b8193f360204 | 966f6d848abe16a4 | d3b2a138232d5d7e | 50% | 5 | 91% | 6.6 cm | 39% |
| chain_L8 | 512 | 2048 | 23.4 | 8f99104477a0a6d3 | 92426ed25efc523d | 37000626381c138e | 75% | 9 | 82% | 5.5 cm | 38% |
| agent | 512 | 2048 | 3.8 | 8a4c0c96c0813f2f | 239c51fc2471eacc | 500b97c8d4c6501c | 36% | 4 | 98% | 5.2 cm | 34% |
agent mode counts (samples): stationary 440, crossing 576, approach 504, oscillating 528.
subsets.json sha256 prefix dc01e7c3dc3a669a; root-list sha256: 1pct 7f64b7df68ea1282 (41 roots/164 samples),
10pct 3a3e2f5813262704 (410/1640), 100pct 19f339fae3cfd285 (4096/16384).
Reproducibility: re-generating v2 train/val/test/w3 with the current code gives byte-identical events.npz,
labels.npz and tensors.pt; only `*_params.jsonl` differs, because EpisodeParams gained new (default-valued) fields
(others, push_body, aim, chain_axis, agent).

## v3 (EXPLORATORY / post-hoc): hashes and measurements
`python -m physmem.data --name v3 --version v3 --base_seed 20261125 --strata ood_fric_0.15,ood_fric_0.9,misspec_velfric,occ_0.5,paddle`
scales.json sha256 prefix 13d8232af460183e (fit on v3 train only).

Slide = distance from push end to t_k+2 s, over pushes that touched the box (cm, p10/25/50/75/90/99).
fs any = fraction of samples with >=1 free-slide horizon. Box at rest by 2.0 s: 100% in every split.
Tipped: 0% except ood_fric_0.9 0.05% and misspec_velfric 2.3%.
| split | roots | sec | tensors.pt | events.npz | labels.npz | slide cm p10/25/50/75/90/99 | fs any | fs per horizon | contact |
|---|---|---|---|---|---|---|---|---|---|
| train | 4096 | 26.2 | f332c2af7191a3d5 | 3c8bc7fd9d8c4bfb | f0c1500f7df4e7af | 0.0/0.0/1.3/3.1/5.6/11.0 | 62.7% | 0/.24/.59/.63/.63/.63 | 91% |
| val | 512 | 3.3 | 605731584908a2fd | 28d352df083aec24 | d4723d015c1f0644 | 0.0/0.0/1.2/3.2/5.5/11.5 | 61.3% | 0/.23/.58/.61/.61/.61 | 91% |
| test | 512 | 3.2 | 2bec710155152233 | 1c3c35150c27750d | ca1e73c96be3a2b5 | 0.0/0.0/1.2/3.1/5.4/10.1 | 61.5% | 0/.22/.58/.61/.61/.61 | 91% |
| ood_fric_0.15 | 512 | 3.5 | 5c3a5c01c73420a6 | 549e2bf64c6827dd | bd93f65bb562037c | 0.7/3.3/6.9/12.6/19.9/32.2 | 92.0% | 0/.27/.84/.92/.92/.92 | 67% |
| ood_fric_0.9 | 512 | 3.3 | c6fe1db1afd174e4 | 4b4038556bcd7daa | e414b913b02b916b | 0.0/0.0/0.5/1.8/3.1/6.1 | 52.2% | 0/.21/.50/.52/.52/.52 | 97% |
| misspec_velfric | 512 | 3.6 | efd2e072d6c88c7c | ff92d2b50c3ae187 | 536f5261b827f077 | 0.0/0.0/0.6/1.8/3.2/7.0 | 0% (by def.) | 0 | 97% |
| occ_0.5 | 512 | 3.6 | 4ff2a95d3c8da6a7 | 5169cd9f3ea12258 | 9d84619037546ae8 | 0.0/0.0/1.2/3.2/5.5/9.5 | 62.2% | 0/.25/.59/.62/.62/.62 | 90% |
| paddle | 512 | 3.6 | 4f34013a87dbe191 | 08dd4d15323c95eb | 59159cec82d1f3d4 | 1.5/2.4/4.1/6.6/9.4/13.7 | 96.6% | 0/.23/.87/.97/.97/.97 | 99% |

Effect of friction on outcomes (simulator finite difference): each of 200 train-distribution episodes was re-simulated
with friction 0.35 and 0.65, everything else identical. Distance between the two final target positions at
t_0+2 s after push 0, in cm (p10/25/50/75/90, mean):
- v2: 0.06/0.11/0.27/0.83/2.14, mean 0.74
- v3: 0.41/0.94/1.9/3.56/5.4, mean 2.46

Slide after release, in cm (p50/p90):
- v2: mu=0.35 0.2/3.7, mu=0.65 0.1/1.7
- v3: mu=0.35 1.9/7.8, mu=0.65 0.7/4.0

Caveat: even at >=0.4 m/s, ~1/3 of sphere pushes end with no free slide. Off-centre (theta, phi) pushes make the
box slip sideways off the pusher during the push, so it has already stopped at push end. The paddle pushes squarely
and gets 97% free-slide coverage. Making v3 "every push slides" would need a narrower phi or a centred contact
point; that was not done because v3 had to stay identical to v2 apart from speed.
