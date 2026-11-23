"""Station layouts shared by task families: bench, mounts for every robot kind, tracker camera."""
from __future__ import annotations

import math

from .scene import BENCH_H, Scene


MOBILE_BENCH_TOP = 0.62      # round 3: lower bench for mobile manipulators (TidyBot arm base at 0.335 m)


def bench_station(scene: Scene, center=(0.0, 0.0), half=(0.6, 0.4), top=BENCH_H, gantry_h=1.45, mu=0.01):
    top = getattr(scene, "bench_top_override", None) or top
    cx, cy = center
    scene.bench(center, half, top=top, mu=mu, name="bench_top")
    standard_mounts(scene, (cx, cy), top, half, gantry_h)
    scene.camera = dict(pos=(cx, cy - 0.55, top + 1.1), lookat=(cx, cy + 0.05, top),
                        view_pos=(cx + 0.9, cy - 1.3, top + 0.95), view_lookat=(cx, cy, top))
    return top


def standard_mounts(scene: Scene, center, top, half, gantry_h=1.45):
    cx, cy = center
    edge = cy - half[1]
    scene.mounts.update({
        "gantry": [((cx, cy, gantry_h), 0.0)],                                   # test-only probe
        "arm": [((cx, edge - 0.05, top), math.pi / 2)],                           # pedestal flush with bench top
        "bimanual": [((cx - 0.25, edge - 0.05, top), math.pi / 2), ((cx + 0.25, edge - 0.05, top), math.pi / 2)],
        "humanoid": [((cx, edge - 0.15, 0.0), math.pi / 2)],                      # floor-standing, facing +y
        "mobile_manipulator": [((cx, edge - 0.25, 0.0), math.pi / 2)],            # base pose on the floor (closer)
    })
