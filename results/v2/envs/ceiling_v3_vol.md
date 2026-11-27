| family | robot | n | nominal | PTA | oracle | oracle - nominal [95% CI] | oracle - PTA [95% CI] | oracle failures | nominal failures | label |
|---|---|---|---|---|---|---|---|---|---|---|
| cable_route | gen3_2f85 | 8 | 1.00 | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |  |  | competence / generalization |
| cable_route | panda | 8 | 0.75 | 1.00 | 0.88 | +0.12 [+0.00, +0.38] | -0.12 [-0.38, +0.00] | crush_or_damage 1 | jam 1, crush_or_damage 1 | competence / generalization |
| cable_route | probe | 8 | 0.88 | 1.00 | 0.88 | +0.00 [+0.00, +0.00] | -0.12 [-0.38, +0.00] | perception 1 | perception 1 | competence / generalization |
| cable_route | ur5e_2f85 | 8 | 0.88 | 1.00 | 0.75 | -0.12 [-0.38, +0.00] | -0.25 [-0.62, +0.00] | jam 1, perception 1 | perception 1 | competence / generalization |
| handover | gen3_2f85 | 8 | 0.88 | 1.00 | 1.00 | +0.12 [+0.00, +0.38] | +0.00 [+0.00, +0.00] |  | tug_or_snatch 1 | competence / generalization |
| handover | panda | 8 | 0.88 | 1.00 | 1.00 | +0.12 [+0.00, +0.38] | +0.00 [+0.00, +0.00] |  | tug_or_snatch 1 | competence / generalization |
| handover | probe | 8 | 0.62 | 0.75 | 0.75 | +0.12 [+0.00, +0.38] | +0.00 [+0.00, +0.00] | slip 1, grasp_miss 1 | grasp_miss 2, tug_or_snatch 1 | competence / generalization |
| handover | ur5e_2f85 | 8 | 0.88 | 1.00 | 1.00 | +0.12 [+0.00, +0.38] | +0.00 [+0.00, +0.00] |  | tug_or_snatch 1 | competence / generalization |
| stack_overhang | gen3_2f85 | 8 | 0.75 | 0.50 | 0.88 | +0.12 [+0.00, +0.38] | +0.38 [+0.12, +0.75] | topple 1 | topple 2 | competence / generalization |
| stack_overhang | panda | 8 | 0.75 | 0.62 | 0.88 | +0.12 [+0.00, +0.38] | +0.25 [+0.00, +0.62] | topple 1 | topple 2 | competence / generalization |
| stack_overhang | probe | 8 | 0.50 | 0.75 | 0.62 | +0.12 [+0.00, +0.38] | -0.12 [-0.38, +0.00] | topple 3 | topple 4 | competence / generalization |
| stack_overhang | ur5e_2f85 | 8 | 0.75 | 0.62 | 0.75 | +0.00 [+0.00, +0.00] | +0.12 [-0.25, +0.50] | topple 2 | topple 2 | competence / generalization |
