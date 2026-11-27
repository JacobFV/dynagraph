"""Splits for v2 environment data (agent ENVS).

* train / val / test: by ROOT SEED (sha256 bucket 80/10/10), decided before any episode is generated, so no
  episode (and no window cut from it) can appear in two splits.
* heldout_layout: test-bucket seeds with the station rotated 180 deg about its centre (source/target sides swap
  relative to the robot) for bench stations; for pallet / conveyor / aisle stations a count shift (more layers /
  items / totes) because a rotation would put the work out of reach.
* heldout_objects: test-bucket seeds with an object set / property range never used in train
  (e.g. convex-mesh + ellipsoid parts in bin_clearing, heavier fixtures, stiffer pouches).
* heldout_family: one family per category is never generated for train/val: bulk -> tote_transfer,
  fine -> connector_insert (their episodes exist only with split="heldout_family").
* heldout_robot: scripts/envs_generate.py --heldout_robot NAME generates that robot ONLY with split="heldout_robot"
  (test-bucket seeds, in-distribution params), never in train/val/test.
"""
from __future__ import annotations

import hashlib

from ..v2.api import TaskSpec

SPLITS = ("train", "val", "test", "heldout_layout", "heldout_objects", "heldout_family", "heldout_robot")
# envs-r6: drawer_open is held out for G5 (zero-shot family transfer; oracle-vs-prior gap +0.30 [+0.10, +0.50] in
# the r6 decision-relevance table; door_open - the closest family - stays in training). Older tranches (r2-r5)
# DO contain drawer_open train/val episodes: exclude them when evaluating G5.
HELDOUT_FAMILIES = {"bulk": "tote_transfer", "fine": "connector_insert", "g5": "drawer_open"}
BENCH_FAMILIES = {"bin_clearing", "singulation", "bulk_sweep", "peg_insert", "connector_insert", "tool_return",
                  "fixture_place", "kitting_pocket", "drawer_open", "door_open", "twist_lock", "machine_tending",
                  "small_parts_pick", "small_peg_insert"}
TRAIN_SHAPES = ("box", "cylinder", "capsule", "sphere")

# params applied to EVERY split of a family (in-distribution object sets)
BASE_PARAMS = {
    "bin_clearing": dict(shapes=TRAIN_SHAPES),
    "small_parts_pick": dict(shapes=TRAIN_SHAPES),
    "singulation": dict(p_box=0.7),
}
HELDOUT_OBJECTS = {
    "bin_clearing": dict(shapes=("mesh", "ellipsoid")),
    "singulation": dict(p_box=0.0),                       # all cylinders (roll/rotate differently when pushed)
    "bulk_sweep": dict(mix=(0.1, 0.2)),                   # mostly capsules (they roll sideways)
    "depalletize": dict(heavy_p=0.6),
    "conveyor_sort": dict(heavy_p=0.6),
    "tote_transfer": dict(content_mass=12.0),
    "peg_insert": dict(peg_half=0.016),
    "connector_insert": dict(),
    "tool_return": dict(head_mass=1.2),
    "fixture_place": dict(mass=5.0),
    "kitting_pocket": dict(n_parts=4),
    "drawer_open": dict(mass=16.0),
    "door_open": dict(mass=8.0),
    "pouch_transfer": dict(young=5e4),
    "twist_lock": dict(frictionloss=0.9),
    "machine_tending": dict(mass=8.0),
    "small_parts_pick": dict(shapes=("mesh", "ellipsoid")),
    "small_peg_insert": dict(peg_half=0.0085),
    "shelf_restock": dict(n_items=7),
    # envs-v3: one hidden quantity pushed just outside its train range (not used by the envs-v3 ceilings)
    "pour": dict(visc=0.08),                      # train 0.001-0.05 N s/m
    "granular_dose": dict(grain_r=0.0075),        # train 4.5-6.5 mm
    "cloth_fold": dict(thickness=0.002),          # train 0.3-1.5 mm
    "cable_route": dict(bend_k=0.3),              # train 0.003-0.2 N m/rad
    "stack_overhang": dict(overhang_m=0.075),     # train 0.055-0.072 m
    "handover": dict(tau=2.5),                    # train 0.3-2.0 s
    "g1_push_terrain": dict(mu_floor=0.06),       # train 0.08-0.9
}
COUNT_SHIFT = {"shelf_restock": dict(n_items=7), "depalletize": dict(layers=3), "conveyor_sort": dict(n_parts=7), "tote_transfer": dict(n_totes=4),
               "pouch_transfer": dict(mass=0.8)}


def seed_bucket(seed: int) -> str:
    h = int.from_bytes(hashlib.sha256(f"physmem-v2-envs|{seed}".encode()).digest()[:4], "little") % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def category_of(family):
    from . import family_class
    return family_class(family).category


def split_params(family, split):
    p = dict(BASE_PARAMS.get(family, {}))
    if split == "heldout_layout":
        if family in BENCH_FAMILIES:
            p["layout"] = "rot180"
        else:
            p.update(COUNT_SHIFT.get(family, {}))
    elif split == "heldout_objects":
        p.update(HELDOUT_OBJECTS.get(family, {}))
    return p


def make_tasks(family, split, n, robot="probe", base_seed=0):
    """Deterministic list of n TaskSpecs for (family, split). Root seeds are drawn from the integers >= base_seed
    whose bucket matches (train/val/test); held-out splits draw from the test bucket."""
    if split in ("train", "val") and family in HELDOUT_FAMILIES.values():
        raise ValueError(f"{family} is a held-out family: only split='heldout_family' (or test-time splits)")
    want = split if split in ("train", "val", "test") else "test"
    out, s = [], base_seed
    while len(out) < n:
        if seed_bucket(s) == want:
            out.append(TaskSpec(family, category_of(family), s, robot, split, split_params(family, split)))
        s += 1
    return out
