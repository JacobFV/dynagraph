"""v2 skills: robot-agnostic parameterized primitives (see docs/ROBOTS.md section 3).

    from physmem.skills import SkillContext, ObjectEstimate, run_skill
    ctx = SkillContext(model, data, emb, on_step=hook)       # reuse ONE ctx per robot per episode (it remembers grasps)
    res = run_skill(ctx, SkillCall("pick", {...}, target="obj3", effector="hand"), estimate)
"""
from physmem.skills.base import ObjectEstimate, SkillContext, SkillRun
from physmem.skills import primitives as P

SKILLS = {'reach': P.reach, 'pick': P.pick, 'place': P.place, 'push': P.push, 'sweep': P.sweep, 'insert': P.insert,
          'open': P.articulate, 'close': P.articulate, 'handover': P.handover}


def run_skill(ctx, call, est=None):
    """dispatch an api.SkillCall. est: ObjectEstimate of call.target (actor-legal), or None where not needed."""
    if getattr(ctx.emb.spec, 'kind', '') == 'humanoid' and call.skill == 'pick' and est is not None:
        return _humanoid_pick(ctx, call, est)
    if call.skill not in SKILLS:
        with SkillRun(ctx, call.skill) as r:
            return r.result(False, 'unsupported')
    return SKILLS[call.skill](ctx, call, est)


def _humanoid_pick(ctx, call, est):
    """G1 + Dex3: pinch pick (default) or two-palm carry (params two_palm=1, e.g. totes 1-3 kg); see humanoid.py."""
    from physmem.embodiments import humanoid as H
    p = call.params
    with SkillRun(ctx, 'pick') as r:
        if p.get('two_palm'):
            info = H.g1_tote_carry(ctx, est.pos, est.half_size, squeeze_n=p.get('squeeze_n', 30.0), lift=p.get('lift', 0.10),
                                   carry_dy=p.get('carry_dy', 0.0), hold_s=p.get('hold_s', 0.0))
            info = dict(info) if isinstance(info, dict) else {}
            st = 'fell' if info.get('fell') else 'dropped' if info.get('dropped') else \
                'done' if info.get('touched', True) else 'no_contact'
            # max_lift is measured from the tote's TRUE pose inside humanoid.py: evaluator-only, not returned to the actor
            return r.result(st == 'done', st, **{k: v for k, v in info.items() if k in ('touched', 'fell', 'dropped')})
        side = call.effector or 'right'
        st = H.g1_pinch_pick(ctx, side, est.pos, est.half_size, lift=p.get('lift', 0.10), guard=p.get('guard', 8.0),
                             squeeze=p.get('squeeze', 1.0))
        st = st if isinstance(st, str) else str(st)
        return r.result(st == 'done', st)


__all__ = ['ObjectEstimate', 'SkillContext', 'run_skill', 'SKILLS']
