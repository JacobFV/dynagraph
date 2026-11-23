"""evaluator-side analysis: can a Dex3 pinch hold each fixture_place seed's part? hold iff 2 mu_grip F >= m (g + a)."""
import numpy as np
from physmem.envs import make_env
from physmem.v2.api import TaskSpec
for seed in range(5):
    env = make_env('fixture_place'); env.reset(TaskSpec('fixture_place', env.category, seed, 'g1_dex3', 'train'))
    h = env.hidden_task
    F = env.robot.emb.max_grip_force('left')
    need = h['mass'] * (9.81 + 1.0)
    cap = 2 * h['mu_grip'] * F
    print(seed, 'mass', round(h['mass'], 2), 'mu_grip', round(h['mu_grip'], 2), 'F_side', round(F, 1), 'capacity', round(cap, 1), 'need', round(need, 1), 'feasible', cap >= need)
