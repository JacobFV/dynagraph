# EXT: external third-person benchmarks and human body/hand data (V3 track EXT; started 2026-11-26)

Owner: agent EXT, branch `ext`. Scope (from `V3_GENERAL.md`): license and access audit, local download (never re-uploaded),
loaders, the official evaluation protocols with the published baselines, a pixels → keypoint/object-track front end for
the V3 third-person track modality, a human hand/body pose front end, and the H6 test design.

Code: `src/physmem/ext/` (`tracks.py` record format, `physion.py` loader, `ocp.py` official readout,
`frontend_color.py` object tracks, `human_pose.py` hand pose, `dexycb.py` loader), `scripts/ext/`, `tests/test_ext.py`.
Data lives only on the local disk under `/home/brandonin/datasets/<name>/`, with sha256 and provenance in
`/home/brandonin/datasets/<name>/manifest.json`, mirrored in `results/ext/datasets_manifest.json`.

## 1. License and access audit (verified 2026-11-26 from official pages; nothing is assumed)

"Reg." means an account, form or license click-through is required before downloading. NS means the official pages do
not state it. H3/H5/H6 refer to the hypotheses in `V3_GENERAL.md`.

| dataset | official URL | license (commercial?) | reg. | size (smallest official subset) | labels | serves | action |
|---|---|---|---|---|---|---|---|
| **Physion v1** | github.com/cogtoolslab/physics-benchmarking-neurips2021 | MIT (repo LICENSE; no separate data license) | no (public S3) | test movies + labels 0.28 GB; readout-fitting HDF5 17–45 GB/scenario; full test HDF5 408 GB | OCP contact label; per-frame object poses/velocities, segmentation, depth, flow, masses, friction (simulator state, evaluator only) | H5 | **downloaded**: test movies, train MP4s, human results; readout-fitting set as a prefix subset (§3) |
| **Physion++** | dingmyu.github.io/physion_v2 | MIT ("All products created as part of this project is shared under the MIT license", NeurIPS'23 D&B supplement) | no (public S3) | readout 2.97 GB, test 3.89 GB, train 29.9 GB, human 0.9 MB | OCP label; per-object **mass, static/dynamic friction, elasticity**, poses, collisions, segmentation | **H5, H3** (property inference is the point of the benchmark) | **downloading** readout + test + human (complete official protocol fits locally) |
| CLEVRER | clevrer.csail.mit.edu | CC0 (README.txt: "Our dataset is under the CC0.") | no | annotations 157 MB (train) / 78 MB (val); videos 12.4 / 6.2 / 6.2 GB | positions, orientations, velocities, collisions, masks, QA; no mass | H5 (collision/counterfactual QA), H3 partial | queued: validation annotations + videos |
| ComPhy | comphyreasoning.github.io | NS (only "Copyright © 2021 ComPhy") | Google Drive, no login | ~8.4 GB (val.json 9 MB) | hidden **mass, charge** (QA) | H3, H5 | **not downloaded**: no license stated; ask the authors |
| PHYRE | github.com/facebookresearch/phyre | Apache-2.0 | no (`pip install phyre`) | simulator package | 2-D task success | H5 (interactive) | later (an environment, not a dataset) |
| IntPhys 2019 | intphys.cognitive-ml.fr | CC BY 4.0 | no | dev 3 GB; total ~250 GB | possible/impossible, depth, masks | H5 (plausibility) | later if time (dev set) |
| IntPhys 2 | github.com/facebookresearch/IntPhys2 | CC BY-NC 4.0, "limited to evaluation purposes" | no (dl.fbaipublicfiles.com, HF ungated) | 1.80 GB | possible/impossible, principle | H5 (eval only) | later if time; evaluation use only |
| Kubric MOVi A–F | github.com/google-research/kubric | code Apache-2.0; **data license NS** | no (public GCS) | smallest MOVi-F 128 px 48 GB | per-object **mass, friction, restitution**, segmentation, depth, flow | H3 | **not downloaded**: data license unstated; PI decision |
| **DexYCB** | dex-ycb.github.io | CC BY-NC 4.0 (research OK, no commercial use) | no (Google Drive links on the site) | 119 GB; one subject 12.4 GB; models 1.46 GB | object 6-D pose, MANO pose, 2-D/3-D hand joints, segmentation, depth (8 RGB-D views) | **H6** (objects are YCB → masses published) | **downloading** subject 01 + models + calibration |
| HO-3D v2/v3 | github.com/shreyashampali/ho3d | academic research only, non-commercial | OneDrive (terms on the repo) | NS (v3: 103k images) | hand (MANO), YCB object pose, depth | H6 | not now (OneDrive; DexYCB covers the same objects) |
| ARCTIC | arctic.is.tue.mpg.de | MPI non-commercial | **yes** (ARCTIC + MANO + SMPL-X accounts) | images 649 GB (meta + raw_seqs 0.3 GB) | MANO, SMPL-X, articulated object pose, contact | H6 | **PI registration** |
| GRAB | grab.is.tue.mpg.de | MPI non-commercial (also bars training commercial AI) | **yes** | NS | SMPL-X body, MANO hands, object pose, per-vertex contact; mocap, no video | H6 (whole-body lifting) | **PI registration** |
| HOI4D | hoi4d.github.io | CC BY-NC 4.0 | OneDrive/Baidu (no form stated) | NS (2.4 M frames) | category-level object pose, MANO, segmentation, depth | H6 | later |
| OakInk2 | huggingface.co/datasets/kelvin34501/OakInk-v2 | CC BY-SA 4.0 (HF tag) | no (HF ungated) | 2.13 TB (single sequences possible) | MANO, SMPL-X, object part transforms | H6 | later (huge; MANO needed to decode hand params) |
| OakInk v1 | oakink.net | CC BY-NC-SA 3.0 (HF tag) | annotations need a **Google form** | NS | MANO, object pose, affordance | H6 | PI form |
| HOT3D | facebookresearch/hot3d | CC BY-SA sequences, CC BY-NC-SA hand data, no-sale clause | **yes** (email sign-up, 14-day links) | Clips on HF 762 GB | MANO/UmeTrack hands, object poses; no mass | H6 | **PI registration** |
| ContactPose | github.com/facebookresearch/ContactPose | data MIT (repo) / CC BY 4.0 (DataPort) | **yes** (IEEE DataPort subscription) | data.zip 7.45 GB; grasps 120 MB | thermal contact maps, hand pose, object pose (static grasps) | H6 (contact) | **PI registration** |
| H2O | h2odataset.ethz.ch | academic, non-commercial | **yes** ("I Agree" + form, 7-day credentials) | NS | 6-D object pose, 3-D hand pose, RGB-D | H6 | **PI registration** |
| YCB objects | ycbbenchmarks.com, registry.opendata.aws/ycb-benchmarks | CC BY 4.0 | no | — | **masses**: Calli et al., IEEE RAM 2015, Table 2 | H6 labels | masses transcribed from the paper (§6) |

Body models and pose estimators (needed for §5):

| component | license (code / weights) | reg. | MANO/SMPL dependency | decision |
|---|---|---|---|---|
| **MediaPipe Hand/Pose Landmarker** | Apache-2.0 / Apache-2.0 (model cards) | no | none | **chosen** (§5) |
| MANO | MPI non-commercial; no redistribution | yes | — | PI registration (only to decode MANO params or run HaMeR/WiLoR) |
| SMPL-X | MPI non-commercial (commercial via Meshcapade) | yes | — | PI registration (ARCTIC/GRAB bodies) |
| HaMeR | MIT / weights NS | needs MANO | yes | candidate after MANO registration (3-D hand mesh) |
| WiLoR | CC BY-NC-ND 4.0 | needs MANO | yes | NC-ND, so not chosen |
| RTMPose / MMPose | Apache-2.0 / weights NS (trained on COCO-WholeBody, Halpe) | no | none | fallback body/hand 2-D |
| ViTPose | Apache-2.0 / weights NS | no | none | fallback |
| Sapiens (Meta) | CC BY-NC 4.0 / CC BY-NC 4.0 | no (HF ungated) | none | NC; not chosen |

### 1a. Registration list for the PI (EXT does not create accounts or accept licenses)
1. **MANO** (mano.is.tue.mpg.de): needed to decode MANO parameters in DexYCB/HO-3D/ARCTIC/GRAB/OakInk and to run
   HaMeR. Non-commercial.
2. **ARCTIC** (arctic.is.tue.mpg.de; also needs the MANO and SMPL-X accounts): bimanual articulated-object
   manipulation with contact. Best data for articulation and handover in H6.
3. **GRAB** (grab.is.tue.mpg.de): whole-body grasp/lift mocap with contact. Mocap, not video, so it suits H6 as
   "perfect tracks".
4. **SMPL-X** (smpl-x.is.tue.mpg.de): body model for ARCTIC/GRAB.
5. **HOT3D** (projectaria.com/datasets/hot3d, email sign-up): egocentric hand-object tracking with PBR object models.
6. **ContactPose** (IEEE DataPort subscription): contact maps.
7. **H2O** (h2odataset.ethz.ch agreement form): two-hand object manipulation, RGB-D.
8. **OakInk v1 annotations** (Google form at oakink.net).

Two license questions for the PI (no registration needed, but terms unclear): **Kubric MOVi** data has no stated
license (it has per-object mass/friction/restitution labels and is ideal for H3), and **ComPhy** states no license.
EXT will not download either until the PI decides or the authors confirm.

## 2. Local downloads (cap 150 GB downloaded; footprint ≤ 300 GB; ~1.5 MB/s link, so prioritised)
Two caps apply, both enforced in `fetch.py`:
- at most 150 GB **downloaded**;
- a total local **footprint** (raw + extracted) of at most 300 GB, set by the lead on 2026-11-26.

Large archives (DexYCB subjects, models) are extracted, verified member by member against the archive, and then
deleted (`scripts/ext/extract_and_prune.py`). Deletion happens only after the archive's sha256 has been re-checked
against the manifest. The manifest keeps URL + sha256 + `deleted_after_extraction`, so any archive can be
re-fetched and checked. The download order is one queue, `scripts/ext/fetch_chain.sh`:
Physion++ → CLEVRER val → DexYCB s01 → Physion v1 readout prefixes (5%) → DexYCB s02–08.
Manifest: `results/ext/datasets_manifest.json` (url, bytes, sha256, license, license source, UTC fetch time).
Scripts: `scripts/ext/fetch.py` (whole official files, resumable) and `scripts/ext/physion_readout_prefix.py`
(prefix subset of the Physion readout archives, §3); `scripts/ext/pget.py` fetches S3 files over 8 parallel
range connections, because S3 gives ~0.2 MB/s per connection here and the link saturates at ~2.3 MB/s aggregate.
Order: Physion v1 test/train MP4s/human results → MediaPipe
model files → Physion++ readout/test/human → DexYCB subject 01 (+ models, calibration) → CLEVRER validation →
Physion v1 readout prefixes (5% of each archive, ~50 stimuli per scenario; resumable, so it can grow).

## 3. Physion v1: loader, official protocol, published baselines

**Data used** (all from the official MIT-licensed S3 bucket `physics-benchmarking-neurips2021-dataset`):
- `Physion.zip` holds the official TEST set: 150 movies per scenario × 8 scenarios, the model-facing
  `mp4s-redyellow` versions (512×512, 30 fps), and `labels.csv` with ground-truth OCP outcomes (600 contact / 600 not).
- READOUT-FITTING set: the official set is 1000 HDF5 stimuli per scenario (256×256 frames stored as PNG inside the
  HDF5, 17–45 GB per scenario). With a ~1.5 MB/s link we fetch a byte **prefix** of each official archive and keep
  the complete members (`scripts/ext/physion_readout_prefix.py`). That gives the first ~5% of stimuli in archive order (about 50 per scenario).
  Archive order mixes sub-configurations (checked on Collide) but is not a random sample.
  **Any number fitted on it is labelled "reduced readout set", which is not the official protocol.**
  The README link for the Roll readout archive (`Rollreadout_HDF5s.tar.gz`) returns 404. The bucket key is
  `Roll_readout_training_HDF5s.tar.gz`.
- `model_human_results.zip`: the official per-participant human responses and model outputs.

**Label rule (official):** `label = any(frame['labels']['target_contacting_zone'] for frame in frames)`, from
`data/extract_labels.ipynb` in the benchmark repo. The agent (red) is `static/target_id` and the patient (yellow) is
`static/zone_id`.

**Official readout protocol, reproduced in `src/physmem/ext/ocp.py`:**
- Sources: neuroailab/physics-models `physion/objective/objective.py`, neuroailab/physopt-physics-benchmarking
  `physopt/config.py` (both MIT), and paper §2.4.
- Readout model: `Pipeline([StandardScaler, LogisticRegression(max_iter=200)])`, `GridSearchCV` over
  `C ∈ logspace(-8, 8, 17)`, `StratifiedKFold(5, shuffle=True, random_state=seed)`, seeds {0, 1, 2}.
- Readouts are fit per scenario on the readout-fitting set and tested on the 150 test movies of that scenario.
- Protocols:
  - **observed**: features of the first 45 frames (1.5 s, the part humans saw);
  - **observed+simulated**: observed features plus the model's predicted future;
  - **full**: the whole movie. This is detection, not prediction, and is not comparable with humans.
- Metrics:
  - accuracy;
  - Pearson r between P(contact) and the mean human response per stimulus;
  - Cohen's κ between the model's binary predictions and each participant, averaged over participants.

  This is our implementation of the definitions in the paper's text.

**Published baselines** (verbatim, `results/ext/physion_published_baselines.json`; Bear et al., NeurIPS 2021 D&B,
arXiv:2106.08261v3, Table S1, accuracy with bootstrapped 95% CIs):

| model | Dominoes | Support | Collide | Contain | Drop | Link | Roll | Drape |
|---|---|---|---|---|---|---|---|---|
| Human | 0.693 | 0.763 | 0.809 | 0.767 | 0.744 | 0.643 | 0.883 | 0.678 |
| SVG | 0.538 | 0.596 | 0.597 | 0.56 | 0.533 | 0.544 | 0.561 | 0.545 |
| OP3 | 0.47 | 0.516 | 0.511 | 0.499 | 0.526 | 0.545 | 0.544 | 0.548 |
| CSWM | 0.471 | 0.691 | 0.552 | 0.557 | 0.577 | 0.627 | 0.609 | 0.55 |
| RPIN | 0.625 | 0.62 | 0.645 | 0.601 | 0.551 | 0.597 | 0.622 | 0.596 |
| pVGG-mlp | 0.601 | 0.669 | 0.651 | 0.638 | 0.606 | 0.614 | 0.573 | 0.6 |
| pVGG-lstm | 0.603 | 0.675 | 0.651 | 0.643 | 0.603 | 0.618 | 0.573 | 0.599 |
| pDEIT-mlp | 0.664 | 0.686 | 0.677 | 0.664 | 0.619 | 0.59 | 0.62 | 0.608 |
| pDEIT-lstm | 0.664 | 0.687 | 0.681 | 0.669 | 0.614 | 0.592 | 0.616 | 0.608 |
| GNS* | 0.604 | 0.695 | 0.85 | 0.652 | 0.708 | 0.73 | 0.735 | 0.653 |
| GNS-R* | 0.591 | 0.686 | 0.842 | 0.683 | 0.712 | 0.725 | 0.792 | 0.653 |
| DPI* | 0.715 | 0.626 | 0.85 | 0.711 | 0.755 | 0.657 | 0.789 | 0.556 |

\* Particle models get ground-truth simulator particles. DPI thresholds a contact distance and fits no readout. The
paper reports overall human accuracy as 0.71. Table S1's caption does not name the training or readout protocol
behind each cell; main-text Fig. 5A shows all-scenarios training with the observed+simulated readout.

**Physion++** (Tung et al., NeurIPS 2023 D&B, arXiv:2306.15668v2, Table 2; `results/ext/physionpp_published_baselines.json`):

| subset | condition | MCVD | DeiT-mlp | ResNet-mlp | VGG-mlp | ALOE | SlotFormer | DPI-Net* |
|---|---|---|---|---|---|---|---|---|
| Overall | unified, w/ property | 54.3 | 53.3 | 52.2 | 51.8 | 51.9 | 52.2 | 53.0 |
| Overall | separate, w/ property | 54.3 | 55.4 | 55.1 | 55.5 | 53.4 | 56.7 | 60.2 |
| Overall | separate, w/o property | 54.3 | 54.1 | 54.8 | 55.1 | 52.5 | 56.2 | 57.5 |
| Overall | separate, fully observed | 59.9 | 63.9 | 61.2 | 56.0 | 55.5 | 65.1 | 92.2 |
| Mass | separate, w/ property | 54.2 | 56.3 | 54.7 | 55.2 | 53.1 | 58.9 | 75.9 |

The per-property rows (friction, elasticity, deformability) are in the JSON. Physion++'s readout-fitting and test sets
are 192 trials (96 matched pairs) per property, and both are small enough to download in full. So **Physion++ is
where we can run the complete official protocol locally**, and it is the natural H3/H5 test of property inference.

## 4. Pixels → object tracks (third-person track modality)

The record format is `src/physmem/ext/tracks.py::TrackSet`:
- frame times;
- per-entity, per-keypoint pixel tracks `uv`, with NaN where the entity was not detected;
- optional depth and xyz;
- detector score;
- intrinsics;
- a provenance string: source file, front-end version and model sha256.

It mirrors the v3-model `CameraTracker` output: `(u, v, d)` detections with NaN, and frame times with drops.
`to_v3_evidence` turns it into the V3 `MOD_TRACK` evidence channels
`(x, v, f=0, contact=0, f_cmd=0, valid)` per entity and axis. It will be re-pointed at the committed V3 contract.
Labels (OCP outcome, simulator masks, mocap) never enter a TrackSet; they are stored beside it as `*.label.json`.

**Tier A (`frontend_color.py`, version `color-v1`).** This tier uses classical HSV colour segmentation of the
agent (red) and the patient (yellow), which the Physion model-facing stimuli paint for the whole movie. It has no
learned weights: provenance is the code plus OpenCV. For each frame and entity it reports the centroid, the bbox
corners and the area of the largest connected component.
- Thresholds were tuned on **readout-fitting** stimuli only (never test). The first bands confused a brown-red floor
  (H 5–8) with the agent (H ≈ 177) and missed the pale-yellow zone (S ≈ 85). The final bands are red
  H ∈ [168, 180] ∪ [0, 3], S ≥ 110, and yellow H ∈ [20, 32], S ≥ 70, V ≥ 150.
- Check against the simulator segmentation, which is evaluator-only (`physion_ocp.py validate`):
  - on 4 Collide readout stimuli, the median centroid error was 0.1–3.7 px (256 px frames) and recall was 100%;
  - the per-scenario table is in `results/ext/physion_frontend_validation_color-v1.json`.

**Tier B (`scripts/ext/sam2_tracks.py`, `sam2-v1`).**
- Model: SAM 2.1 hiera-small, code and checkpoint Apache-2.0, checkpoint sha256 `6d1aa6f3…`.
- Prompts: automatic masks on the prompt frame (frame 0, or the Physion++ cue frame). For Physion++, the official
  cue map supplies the agent and patient masks.
- Propagation: forward and backward over every 3rd frame (10 fps), up to 6 objects.
- Runs on gb10 under `gpu_small.sh` with peak allocator memory of 2.5 GB.
- Throughput on the shared GPU: ~0.6 s per sampled frame, so ~35–100 s per video.

Check against the simulator segmentation (`_id.mp4`, evaluator-only) on 5 Physion++ readout trials:
- agent: median centroid error 0.3–0.7 px, recall 99–100%;
- patient: median centroid error 0.3–0.8 px; recall 100% in 4 trials and 46% in 1 (lost after an occlusion).

Because throughput is limited, the Physion++ order is mass first (readout 200, then the official test 192). The
other properties follow as gb10 time allows (`scripts/ext/gb10_sam2_driver.sh`, one chunk of 40 videos per lock
acquisition).

## 5. Human hand/body pose front end

**Choice: MediaPipe Hand Landmarker** (21 keypoints per hand, plus 3-D "world" keypoints) and Pose Landmarker (33
body keypoints). Code is Apache-2.0 and the model files are Apache-2.0 per their model cards. There is no MANO/SMPL
dependency, no registration, and it runs on CPU. The model files are fetched by `fetch.py` (`hand_landmarker.task`
sha256 `fbc2a300…`, `pose_landmarker_full.task` sha256 `4eaa5eb7…`).

The 3-D hand-mesh estimators (HaMeR, WiLoR) need MANO, and WiLoR is NC-ND. HaMeR can come in once the PI registers MANO.

`human_pose.py::HandTracker` returns a `TrackSet(kind="human")` with entities `left_hand`/`right_hand` and the
21 MediaPipe joint names. The joint order matches DexYCB/MANO, and the thumb names map index-for-index. With an
aligned depth image and intrinsics, `lift_depth` lifts each keypoint to camera-frame xyz from the median depth in a
5×5 window: an actor-legal RGB-D lift.

Validation on DexYCB (no registration, CC BY-NC 4.0) is in §5a, filled from `results/ext/dexycb_pose_validation.json`.

## 6. H6 test design and pre-registration DRAFT (the lead commits the final text on its own, before any test run)

**Status: DRAFT, not frozen.** Nothing in this section has been run on TEST data.

**Freeze rule (lead decision, 2026-11-26).** EXT freezes this section in its own commit on branch `ext`, with the
message `H6 pre-registration, frozen before any test run`. That happens after V3-MODEL's human-track model and its
generic-memory twin have been trained and run on TRAIN/VAL subjects only, and before any inference on TEST subjects
07–08. **No model, feature extractor or readout is run on subjects 07–08 before that commit.** The download includes
them, and `h6_features.py` must be called with `--subjects` limited to 01–06 until then.

**Question.** Humans are treated as an embodiment without force sensing. From third-person RGB-D video of a person
grasping and lifting an object, does the V3 model with typed memory infer the object's **mass** more accurately than
the same model with generic memory?

**Fragility is out of scope for this test.** We found no human-object dataset accessible without registration that has
fragility or crush labels, and the registered ones do not have them either. Fragility stays untested and is reported
as such.

**Data.** DexYCB (CC BY-NC 4.0). 10 subjects × 100 sequences. Each sequence: one hand grasps one of 20 YCB objects,
5 trials per object per subject, recorded on 8 calibrated RGB-D views at 30 fps.
- Mass labels: nominal YCB masses from Calli et al. 2015, Table 2 (`dexycb.py::YCB_MASS_G`). They span 15.8 g
  (large marker) to 1131 g (bleach cleanser). These are evaluator-only.
- Known limitation: these are catalogue masses, not weighed in DexYCB's lab.
- Splits (fixed now; unseen-subject split like the toolkit's `s1`):
  - TRAIN subjects 01–05;
  - VAL subject 06;
  - TEST subjects 07–08 (200 sequences).
- Subjects 09–10 are not used, which keeps the local download ≤ 150 GB (8 subjects ≈ 99 GB).
- One camera view, fixed now: serial index 0 (`836212060125`), all frames.

**Observations (actor-legal only).**
- The MediaPipe hand TrackSet from §5 (front end `mediapipe-hand-v1`, model sha256 frozen in the TrackSet source),
  lifted to 3-D with the aligned depth and the camera intrinsics (the same lifting a real RGB-D rig would do).
- The MediaPipe world landmarks.
- Not used: DexYCB MANO/joint labels, object 6-D poses, segmentation, the object identity, or any pixel crop of the
  object.
- Not used in the confirmatory test: an object track. There is no off-the-shelf object tracker validated on
  DexYCB yet. A SAM-2-based object track may be added as a separately pre-registered secondary input.

**Arms** (matched observations, labels, readout and budget):
- **TYPED**: the V3 general model, frozen at a commit hash recorded in the frozen pre-registration. It consumes the
  human TrackSet as an embodiment without force sensing. We read out its typed **mass** slot (z and posterior sd)
  for the grasped entity.
- **GENERIC**: the matched generic-memory ablation from V3-MODEL (same backbone, data and training budget). Its
  pooled entity features feed the readout.
- Readout, identical for both arms: ridge regression to log-mass. TYPED uses the 2-D vector [z, log sd]; GENERIC uses
  its feature vector, with PCA to ≤ 16 dims fit on TRAIN. Ridge α is chosen by grouped CV on TRAIN, grouped by object.
  The readout is fit on TRAIN subjects **with leave-objects-out**: 4 object folds (objects with
  `object_index % 4 == k`), so every TEST prediction comes from a readout that never saw that object.
- **Controls** (reported; they are not the confirmatory comparison):
  - C0: train-mean log-mass;
  - C1: grasp-aperture-only. Static hand features at grasp onset (thumb-index distance, hand bbox), the same ridge
    readout. This measures how much "mass" is just object size seen through the grasp.
  - C2: a classical kinematic estimator, as the V3 §0 item 5 comparator. Hand-crafted lift features (time from grasp
    onset to lift-off, peak vertical velocity and acceleration of the wrist, lift duration) with the same ridge
    readout.
  - C3: object-identity lookup. Map the visual object class to the mean TRAIN mass of that class. This documents
    the "recognise the object and recall its mass" shortcut. Under leave-objects-out, no TEST object is ever in its
    lookup table, so it falls back to the TRAIN mean (= C0) by construction. It is reported to make that explicit.
    The class comes from the dataset's object label, so this control is a diagnostic with privileged input, not an
    actor-legal arm. (Added at the lead's request, 2026-11-26.)
  - Grasp onset and lift-off are computed from the actor-legal wrist track in `scripts/ext/h6_features.py`
    (`h6-feat-v1`):
    - "up" is the normal of the table plane, RANSAC-fitted to the frame-0 depth cloud (depth and intrinsics only);
    - wrist height h(t) is median-filtered over 5 frames;
    - t_min is the minimum-height frame;
    - lift-off is the first frame after t_min with h − h(t_min) > 2 cm, sustained for 5 frames;
    - grasp onset is the frame of minimum wrist speed in [t_min − 15, lift-off].

    On the first TRAIN sequence, the depth-lifted wrist height follows the DexYCB label to about 1 cm over the
    whole lift. This was a pipeline debug on a TRAIN subject; no mass labels were looked at.

**Primary endpoint.** Per TEST sequence, absolute log-mass error `e = |log m̂ − log m|`. The paired difference is
`d_i = e_GENERIC − e_TYPED` (n = 200 sequences, n ≥ 30 holds).
- **H6 is supported iff the 95% cluster-bootstrap CI of mean d excludes 0 on the positive side.**
- The bootstrap resamples the 20 objects with replacement, then sequences within each object: 10,000 resamples,
  seed 0.
- Objects are the clustering unit because masses are per object. The effective n is therefore 20 objects, and this
  is stated up front as the main power limitation; a null is reportable.

**Secondary (descriptive, no claims).**
- Spearman ρ between per-object mean predicted and true mass, for each arm.
- TYPED posterior calibration: coverage of the 50% and 90% intervals after the affine readout.
- Each arm against C1 and C2: same paired statistic.
- The share of the TYPED-vs-GENERIC gain that remains after partialling out C1's prediction.

**Exclusions.** Decided on actor-legal signals only and applied identically to all arms: a sequence is excluded if
the hand is detected in < 50% of frames, or if lift-off is never detected. Exclusion counts are reported per arm
(they are the same by construction).

**Tuning and freezing.** All choices are tuned on VAL subject 06 only: readout α grid, PCA dims and the front-end
thresholds. Frozen before TEST:
- model and code commit hashes;
- the front-end version and model sha256;
- the view;
- the feature definitions in this section;
- the analysis script `scripts/ext/h6_eval.py` (to be written; it reads TEST only after the freeze commit).

**What would falsify our reading.**
- If C1 (grasp aperture) explains as much as TYPED, the "mass inference" is size inference. We will say so.
- If TYPED beats GENERIC only on in-distribution objects (VAL) and not on held-out objects, the claim fails.

### 3a. Pipeline check: our track-feature reference readout (NOT a model result, NOT the official protocol)
This check runs the official readout (`ocp.py`) on fixed hand-made features of the tier-A colour tracks
(`frontend_color.ocp_track_features`):
- **observed**: centroid, bbox, velocity, distance and gap of agent/patient over frames 0–44;
- **+simulated**: constant-velocity extrapolation of both centroids to 0.5/1/2/4 s;
- **full**: the same features over the whole movie.

The readout-fitting set is the reduced prefix set. Its purpose is only to show that the loader → tracks → readout
→ metrics chain works end to end and to give a floor that any learned model must beat.

Sanity check of the official human data and the label join: recomputing human accuracy from the official
per-participant CSVs (`excluded_games.csv` removed; 794 participants) gives
Dominoes 0.692 / Support 0.755 / Collide 0.803 / Contain 0.760 / Drop 0.740 / Link 0.638 / Roll 0.878 /
Drape 0.677. Each is within 0.008 of Table S1. The paper reports 688 participants kept after exclusions, so the
released exclusion list looks incomplete.

Update 2026-11-27: all 8 scenarios. Reduced readout sets of 47–53 stimuli per scenario; 3 seeds; official test
(150 per scenario). Accuracy:

| scenario | n readout | observed | observed+simulated | full (detection) | published pDEIT-lstm | human |
|---|---|---|---|---|---|---|
| Dominoes | 47 | 0.482 | 0.500 | 0.900 | 0.664 | 0.693 |
| Support | 51 | 0.538 | 0.544 | 0.902 | 0.687 | 0.763 |
| Collide | 50 | 0.538 | 0.562 | 0.720 | 0.681 | 0.809 |
| Contain | 53 | 0.518 | 0.516 | 0.549 | 0.669 | 0.767 |
| Drop | 50 | 0.544 | 0.542 | 0.740 | 0.614 | 0.744 |
| Link | 48 | 0.500 | 0.580 | 0.613 | 0.592 | 0.643 |
| Roll | 51 | 0.518 | 0.522 | 0.673 | 0.616 | 0.883 |
| Drape | 48 | 0.531 | 0.522 | 0.573 | 0.608 | 0.678 |

Colour-tracker check against the simulator masks (20 readout stimuli per scenario,
`results/ext/physion_frontend_validation_color-v2.json`):
- the agent's median centroid error is 0.4–0.7 px with recall 0.94–1.00 in every scenario;
- the patient is good in Support, Collide, Roll and Link (0.4–2.6 px);
- the patient is poor in Dominoes (6 px), Drop (23 px) and Contain (52 px);
- the patient is almost never found in Drape (recall 0.07; the zone there is not a saturated yellow).

The weak "full" readouts in Contain, Drape and Roll track those front-end failures.

Reading: with only the agent and patient tracked, nothing in the first 1.5 s predicts Dominoes outcomes, which
depend on the untracked middle dominoes, so the result is at chance. The full-movie readout detects the contact at
0.90, so the chain works. Rows for the other scenarios are added as their readout prefixes arrive
(`results/ext/physion_ocp_color-v2.json`, `scripts/ext/physion_table.py`).

### 5a. Validation of the hand front end on DexYCB (subject 01, partial download; evaluator labels only)

The run (`scripts/ext/dexycb_pose_validate.py`, `results/ext/dexycb_pose_validation.json`) covered:
- 9 sequences from the first part of the subject-01 archive, all 8 views (65 complete views, 4456 frames with the
  labelled hand present);
- 6 objects: cracker box, potted meat can, bleach cleanser, bowl, scissors, foam brick;
- 48 right-hand and 17 left-hand views.

The labels are DexYCB's mocap-fitted MANO joints (`joint_2d`, `joint_3d`).

| metric | value |
|---|---|
| detection recall (frames with the hand present) | 0.877 (per view 0.55–1.00; the H6 view `836212060125` 0.95) |
| 2-D joint error, median / mean | 7.1 px / 12.1 px (640×480) |
| PCK@10 px / PCK@20 px | 0.664 / 0.895 |
| depth-lifted 3-D error, root-relative, median / mean | 27.4 mm / 63.5 mm (mean inflated by depth outliers at occluded joints) |
| depth-lifted 3-D error, absolute camera frame, median | 24.1 mm |
| MediaPipe world landmarks, root-relative, median | 48.7 mm (no alignment) |
| frames with a valid depth lift | 0.983 |

These are our measurements, not a published benchmark protocol: there is no hand-cropping, MediaPipe runs in
single-hand VIDEO mode, and the thumb-joint names are mapped index for index.

Known biases:
- Depth lifting reads the hand *surface*, while the labels are joint centres, so the absolute 3-D error has a
  floor of roughly 10 mm.
- Under grasp occlusion, the finger joints pick up the object's depth.

For H6 the depth-lifted **wrist** is what matters. On the debug sequence, its height above the RANSAC table plane
follows the label to about 1 cm through the whole lift (§6).

### 3b. Physion++ (complete official sets local; MIT)
- **Readout-fitting set:** 800 trials, 200 per property: mass, friction, elasticity (bouncy), deformability.
  The paper says 192 per property; the archive holds 200.
- **Test set:** `official_test_ids()` reconstructs the official 192 stimuli per property from the official human
  data (`human_data.zip`, the "run2" files: 50 participants × 192 stimuli per property, paper §4.2). This is needed
  because the per-property test CSVs the paper mentions are not linked from the project page.
- **Labels:** `static/does_target_contact_zone`. The observed window is `[0, start_frame_for_prediction)`.
- **Videos are not red/yellow painted.** Agent and patient are given by the official cue map (`*_map.png`). We
  checked that the map is aligned with frame `start_frame_for_prediction`: IoU with the `_id` segmentation of the
  cued object peaks at that frame in 3 of 5 sampled trials and within 3 frames in a fourth.
- **Tracking:** the colour front end cannot be used here. Tracks come from tier B: SAM 2.1 seeded with the cue-map
  masks at the cue frame, propagated both ways, plus automatic masks for the other objects.
- **Pipeline:** `scripts/ext/physionpp_ocp.py` uses the same official readout (`ocp.py`), per property.
- **Hidden properties** (`star_mass`, `star_deform`, friction, bounciness) are in the pkl. They are evaluator-only
  and make Physion++ the natural real-benchmark check for H3.

Tier-B compute note. In the first gb10 smoke run, SAM 2.1-small ran at stride 1 at ~2 frames/s and took 7.2 GB of
GPU, over the 6 GB rule. We killed it within 2 minutes. It now runs with frames and state offloaded to the CPU, a
4.5 GB allocator cap and stride 3 (10 fps).

## 7. Status and running jobs (2026-11-26, 21:00 CST)

**Running unattended.**
- Local download chain `scripts/ext/fetch_chain.sh` (PID in `/home/brandonin/datasets/logs/chain.pid`, log `chain.log`):
  1. CLEVRER val;
  2. DexYCB s01, then extract + prune;
  3. Physion v1 readout prefixes (5%);
  4. DexYCB s02–08, each extracted + pruned.
- gb10 SAM2 drivers, one `gpu_small.sh` chunk of 40 videos at a time:
  - Physion++ mass readout (200), then mass official test (192);
  - logs at `~/dynagraph_ext/logs/sam2_pp_*.log`;
  - PIDs in `/home/brandonin/datasets/logs/gb10_pids.txt`.
- Current gb10 footprint: ~0.7 GB.

**Next steps, in order.**
1. Collect the Physion++ results:
   ```
   rsync -a gb10-direct:dynagraph_ext/tracks_pp/ /home/brandonin/datasets/physionpp/tracks/sam2-v1/
   python -I scripts/ext/physionpp_ocp.py readout
   ```
   Then queue friction, elasticity and deformability: `pp_readout_jobs.txt` / `pp_test_jobs.txt` are already on gb10.
2. As the Physion v1 prefixes land:
   ```
   python -I scripts/ext/physion_ocp.py track
   python -I scripts/ext/physion_ocp.py readout
   python -I scripts/ext/physion_table.py
   ```
   Run tier B on Physion v1 Dominoes, where the untracked middle dominoes are the bottleneck.
3. DexYCB: re-run `dexycb_pose_validate.py` on subjects 01–06 when they are extracted. Run `h6_features.py` on
   01–06 only. TEST subjects 07–08 stay untouched until the H6 freeze commit (§6).
4. CLEVRER: track validation videos with tier B, and use `clevrer.py` annotations as evaluator labels for a derived,
   non-official collision-prediction check.

**Update 2026-11-27: full subject 01, H6 view `836212060125`, all 100 sequences (20 objects × 5)** — results in
`results/ext/dexycb_pose_validation_s01_view0.json`:

| metric | value |
|---|---|
| frames with the hand present | 6866 |
| recall | 0.833 |
| 2-D error, median | 7.0 px |
| PCK@10 / PCK@20 | 0.654 / 0.888 |
| depth-lifted error, root-relative, median | 33.2 mm |
| depth-lifted error, absolute, median | 21.9 mm |
| MediaPipe world-landmark error, root-relative, median | 26.8 mm |

This view is a TRAIN subject; subjects 07–08 were not touched.
