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

## 2. Local downloads (cap 150 GB; ~1.5 MB/s link, so prioritised)
Manifest: `results/ext/datasets_manifest.json` (url, bytes, sha256, license, license source, UTC fetch time).
Scripts: `scripts/ext/fetch.py` (whole official files, resumable) and `scripts/ext/physion_readout_prefix.py`
(prefix subset of the Physion readout archives, §3). Order: Physion v1 test/train MP4s/human results → MediaPipe
model files → Physion++ readout/test/human → DexYCB subject 01 (+ models, calibration) → CLEVRER validation →
Physion v1 readout prefixes (10%, extendable to 20%).
