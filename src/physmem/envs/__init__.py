"""v2 industrial environments (agent ENVS). See docs/ENVS.md.

    from physmem.envs import make_env, FAMILIES
    env = make_env("bin_clearing")
    ev = env.reset(TaskSpec("bin_clearing", "bulk", seed=0, robot="probe", split="train"))
    out = env.execute(env.scripted())
"""
from __future__ import annotations

import importlib

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
}

FAMILIES = tuple(_REG)


def family_class(name):
    mod, cls = _REG[name]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)


def make_env(family, robot_factory=None):
    return family_class(family)(robot_factory=robot_factory)


def register(name, module, cls):
    _REG[name] = (module, cls)
