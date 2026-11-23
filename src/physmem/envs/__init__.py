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
ENV_VERSION = "envs-r5-dev"

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
}

FAMILIES = tuple(_REG)


def family_class(name):
    mod, cls = _REG[name]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)


def make_env(family, robot_factory=None):
    return family_class(family)(robot_factory=robot_factory)


def register(name, module, cls):
    _REG[name] = (module, cls)
