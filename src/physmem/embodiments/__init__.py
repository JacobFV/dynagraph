"""v2 embodiments: one interface for every robot (see docs/ROBOTS.md).

    import physmem.embodiments as E
    emb = E.make("ur5e_2f85", prefix="r0")        # api.Embodiment
    spec = mujoco.MjSpec(); E.scene_defaults(spec)  # timestep 2 ms, implicitfast, elliptic cones
    emb.build(spec, mount_pose=((0, 0, 0), (1, 0, 0, 0)))
    model = spec.compile(); data = mujoco.MjData(model)
    emb.bind(model, data); emb.reset(model, data)
    ctrl = emb.controller()                         # api.Controller
    ...  every physics step:  ctrl.apply(model, data); mujoco.mj_step(model, data)
"""
from physmem.embodiments.core import ASSETS, EffectorCfg, GripperCfg, MjEmbodiment, RobotCfg, scene_defaults
from physmem.embodiments.robots import ROBOTS, names
from physmem.embodiments.trajectory import BlendedTrajectory, make_traj


def make(name: str, prefix: str = 'r0', **kw):
    """construct an embodiment by registry name (see names())."""
    if name in ROBOTS:
        cfg = ROBOTS[name]
        if kw.get('grip_force') is not None:   # per-side squeeze in N (calibrated via GripperCfg.pad_per_drive)
            import dataclasses
            effs = tuple(dataclasses.replace(e, gripper=dataclasses.replace(e.gripper, grip_n=float(kw['grip_force'])))
                         for e in cfg.effectors)
            cfg = dataclasses.replace(cfg, effectors=effs)
        return MjEmbodiment(cfg, prefix=prefix)
    if name.startswith('g1_dex3'):
        from physmem.embodiments.humanoid import G1Embodiment
        return G1Embodiment(prefix=prefix, fixed_base=name.endswith('_fixed'), **kw)
    raise KeyError(f'unknown embodiment {name!r}; known: {names()}')


__all__ = ['make', 'names', 'ROBOTS', 'scene_defaults', 'MjEmbodiment', 'RobotCfg', 'EffectorCfg', 'GripperCfg',
           'BlendedTrajectory', 'make_traj', 'ASSETS']
