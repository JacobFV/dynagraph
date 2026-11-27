"""v2 industrial environments (agent ENVS). See docs/ENVS.md.

    from physmem.envs import make_env, FAMILIES
    env = make_env("bin_clearing")
    ev = env.reset(TaskSpec("bin_clearing", "bulk", seed=0, robot="probe", split="train"))
    out = env.execute(env.scripted())
"""
from __future__ import annotations

import importlib

# environment version, recorded in every generation manifest. envs-r2: tranches 1-3 of the large run;
# envs-r3: conveyor sorted-item memory, stiff pad pairs, payload-aware depalletize + 'handover' flag, slick
# cartons, mobile lower bench, light-parts station (small_parts_pick, small_peg_insert).
# envs-r4: fragile parts (hidden crush limits, pad-squeeze damage, solved() = success and no damage), snapshot /
# restore / set_hidden / resimulate, bin_clearing 3-6 parts, singulation bounded work area, peg cycle-time limit.
# envs-r5 (dev): peg spiral time cost + dent limit, pouch squeeze limit, singulation pick option + cycle limit,
# shelf_restock (mobile aisle with move_base), generator flushes buffers on SIGTERM.
# envs-r6: site dressing (params["dressing"] = "site", envs/dressing.py) after the TalOS warehouse.
# envs-v3: new domains (cable, cloth, pour, granular scoop, stacking, G1 terrain, handover); older families unchanged.
ENV_VERSION = "envs-v3"

_REG = {
    "bin_clearing": ("families_bulk", "BinClearing"),
    "singulation": ("families_bulk", "Singulation"),
    "bulk_sweep": ("families_bulk", "BulkSweep"),
    "peg_insert": ("families_fine", "PegInsert"),
    "connector_insert": ("families_fine", "ConnectorInsert"),
    "tool_return": ("families_fine", "ToolReturn"),
    "fixture_place": ("families_fine", "FixturePlace"),
    "kitting_pocket": ("families_fine", "KittingPocket"),
    "drawer_open": ("families_fine", "DrawerOpen"),
    "door_open": ("families_fine", "DoorOpen"),
    "depalletize": ("families_logistics", "Depalletize"),
    "conveyor_sort": ("families_logistics", "ConveyorSort"),
    "tote_transfer": ("families_logistics", "ToteTransfer"),
    "pouch_transfer": ("families_logistics", "PouchTransfer"),
    "twist_lock": ("families_extra", "TwistLock"),
    "machine_tending": ("families_extra", "MachineTending"),
    "small_parts_pick": ("families_extra", "SmallPartsPick"),
    "small_peg_insert": ("families_extra", "SmallPegInsert"),
    "shelf_restock": ("families_extra", "ShelfRestock"),
    "tight_insert": ("families_fine_motor", "TightInsert"),         # envs-r7 fine motor
    "fastener_pick": ("families_fine_motor", "FastenerPick"),       # envs-r7 fine motor
    # envs-v3 (agent ENVS-V3): new physical domains for the V3 general model (docs/ENVS.md "envs-v3")
    "pour": ("families_v3_media", "Pour"),
    "granular_scoop": ("families_v3_media", "GranularScoop"),
    "cloth_fold": ("families_v3_deform", "ClothFold"),
    "cable_route": ("families_v3_deform", "CableRoute"),
    "stack_overhang": ("families_v3_rigid", "StackOverhang"),
    "handover": ("families_v3_rigid", "Handover"),
    "g1_push_terrain": ("families_v3_g1", "G1PushTerrain"),
}

FAMILIES = tuple(_REG)


def family_class(name):
    mod, cls = _REG[name]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)


def make_env(family, robot_factory=None):
    return family_class(family)(robot_factory=robot_factory)


def register(name, module, cls):
    _REG[name] = (module, cls)
