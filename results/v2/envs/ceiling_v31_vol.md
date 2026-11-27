| family | robot | n | nominal | PTA | oracle | oracle - nominal [95% CI] | oracle - PTA [95% CI] | oracle failures | nominal failures | label |
|---|---|---|---|---|---|---|---|---|---|---|
| cable_route | gen3_2f85 | 20 | 0.90 | 0.90 | 0.90 | +0.00 [-0.15, +0.15] | +0.00 [-0.15, +0.15] | jam 2 | jam 1, crush_or_damage 1 | competence / generalization |
| cable_route | panda | 20 | 0.55 | 0.65 | 0.60 | +0.05 [-0.10, +0.20] | -0.05 [-0.20, +0.10] | crush_or_damage 5, jam 2, slip 1 | crush_or_damage 6, jam 2, slip 1 | competence / generalization |
| cable_route | probe | 20 | 0.70 | 0.75 | 0.75 | +0.05 [+0.00, +0.15] | +0.00 [-0.15, +0.15] | jam 4, perception 1 | jam 3, crush_or_damage 2, perception 1 | competence / generalization |
| cable_route | ur5e_2f85 | 20 | 0.85 | 0.90 | 0.80 | -0.05 [-0.15, +0.00] | -0.10 [-0.25, +0.00] | jam 3, perception 1 | perception 1, jam 1, crush_or_damage 1 | competence / generalization |
| cloth_fold | gen3_2f85 | 20 | 0.45 | - | 0.55 | +0.10 [+0.00, +0.25] | - | slip 9 | slip 11 | competence / generalization |
| cloth_fold | panda | 20 | 0.00 | - | 0.00 | +0.00 [+0.00, +0.00] | - | grasp_miss 18, perception 2 | grasp_miss 18, perception 2 | competence / generalization |
| cloth_fold | probe | 20 | 0.55 | - | 0.50 | -0.05 [-0.30, +0.20] | - | place_miss_or_predicate 8, perception 2 | place_miss_or_predicate 8, perception 1 | competence / generalization |
| cloth_fold | ur5e_2f85 | 20 | 0.50 | - | 0.50 | +0.00 [+0.00, +0.00] | - | slip 7, perception 2, grasp_miss 1 | slip 7, perception 2, grasp_miss 1 | competence / generalization |
| g1_push_terrain | g1_dex3 | 20 | 0.20 | 0.10 | 0.15 | -0.05 [-0.15, +0.00] | +0.05 [-0.10, +0.20] | foot_slip 9, jam 4, perception 2, collision 2 | collision 7, foot_slip 6, perception 2, jam 1 | competence / generalization |
| g1_push_terrain | g1_dex3_fixed | 20 | 1.00 | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |  |  | competence / generalization |
| handover | gen3_2f85 | 20 | 0.75 | 1.00 | 1.00 | +0.25 [+0.10, +0.45] | +0.00 [+0.00, +0.00] |  | tug_or_snatch 5 | decision-relevant |
| handover | panda | 20 | 0.65 | 0.85 | 0.80 | +0.15 [-0.05, +0.35] | -0.05 [-0.20, +0.10] | collision 2, slip 1, tug_or_snatch 1 | tug_or_snatch 6, slip 1 | decision-relevant |
| handover | probe | 20 | 0.35 | 0.70 | 0.75 | +0.40 [+0.20, +0.60] | +0.05 [+0.00, +0.15] | grasp_miss 3, slip 2 | grasp_miss 6, tug_or_snatch 4, slip 2, place_miss_or_predicate 1 | decision-relevant |
| handover | ur5e_2f85 | 20 | 0.55 | 1.00 | 1.00 | +0.45 [+0.25, +0.65] | +0.00 [+0.00, +0.00] |  | tug_or_snatch 9 | decision-relevant |
| pour | gen3_2f85 | 20 | 0.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | place_miss_or_predicate 20 | place_miss_or_predicate 20 | decision-relevant |
| pour | panda | 20 | 0.05 | 0.10 | 0.05 | +0.00 [-0.15, +0.15] | -0.05 [-0.15, +0.00] | place_miss_or_predicate 11, slip 5, grasp_miss 3 | place_miss_or_predicate 11, slip 5, grasp_miss 3 | decision-relevant |
| pour | probe | 20 | 0.20 | 0.65 | 0.80 | +0.60 [+0.40, +0.80] | +0.15 [-0.05, +0.35] | place_miss_or_predicate 4 | place_miss_or_predicate 16 | decision-relevant |
| pour | ur5e_2f85 | 20 | 0.60 | 0.55 | 0.55 | -0.05 [-0.30, +0.20] | +0.00 [-0.20, +0.20] | place_miss_or_predicate 9 | place_miss_or_predicate 8 | decision-relevant |
| stack_overhang | gen3_2f85 | 20 | 0.45 | 0.20 | 0.60 | +0.15 [+0.00, +0.30] | +0.40 [+0.15, +0.65] | place_miss_or_predicate 5, topple 3 | topple 11 | decision-relevant |
| stack_overhang | panda | 20 | 0.20 | 0.05 | 0.45 | +0.25 [+0.10, +0.45] | +0.40 [+0.20, +0.60] | topple 8, place_miss_or_predicate 3 | topple 15, perception 1 | decision-relevant |
| stack_overhang | probe | 20 | 0.15 | 0.45 | 0.55 | +0.40 [+0.20, +0.60] | +0.10 [-0.15, +0.35] | place_miss_or_predicate 5, topple 4 | topple 15, perception 2 | decision-relevant |
| stack_overhang | ur5e_2f85 | 20 | 0.15 | 0.15 | 0.50 | +0.35 [+0.15, +0.55] | +0.35 [+0.10, +0.60] | place_miss_or_predicate 5, topple 4, perception 1 | topple 15, perception 2 | decision-relevant |
