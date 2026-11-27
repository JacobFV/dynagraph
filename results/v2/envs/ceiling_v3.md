success (mean progress) over seeds; site-dressed; tag v3

| robot | family | nominal | probe_then_act | oracle | oracle failure causes | nominal failure causes | median success sim-s (nominal / probe_then_act / oracle) |
|---|---|---|---|---|---|---|---|
| gen3_2f85 | cable_route | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |  |  | 24.1 / 32.9 / 24.1 |
| gen3_2f85 | handover | 0.88 (0.93) | 1.00 (1.00) | 1.00 (1.00) |  | tug_or_snatch 1 | 14.0 / 17.4 / 14.9 |
| gen3_2f85 | stack_overhang | 0.75 (0.84) | 0.50 (0.70) | 0.88 (0.93) | topple 1 | topple 2 | 47.2 / 76.9 / 47.3 |
| panda | cable_route | 0.75 (0.94) | 1.00 (1.00) | 0.88 (1.00) | crush_or_damage 1 | jam 1, crush_or_damage 1 | 24.5 / 37.6 / 24.5 |
| panda | handover | 0.88 (0.93) | 1.00 (1.00) | 1.00 (1.00) |  | tug_or_snatch 1 | 13.3 / 16.6 / 14.3 |
| panda | stack_overhang | 0.75 (0.84) | 0.62 (0.80) | 0.88 (0.90) | topple 1 | topple 2 | 45.7 / 74.7 / 45.9 |
| probe | cable_route | 0.88 (0.88) | 1.00 (1.00) | 0.88 (0.88) | perception 1 | perception 1 | 24.8 / 43.2 / 24.8 |
| probe | handover | 0.62 (0.77) | 0.75 (0.89) | 0.75 (0.89) | slip 1, grasp_miss 1 | grasp_miss 2, tug_or_snatch 1 | 13.3 / 17.2 / 14.5 |
| probe | stack_overhang | 0.50 (0.68) | 0.75 (0.93) | 0.62 (0.74) | topple 3 | topple 4 | 45.8 / 74.6 / 45.8 |
| ur5e_2f85 | cable_route | 0.88 (0.88) | 1.00 (1.00) | 0.75 (0.81) | jam 1, perception 1 | perception 1 | 24.9 / 33.7 / 25.1 |
| ur5e_2f85 | handover | 0.88 (0.93) | 1.00 (1.00) | 1.00 (1.00) |  | tug_or_snatch 1 | 14.0 / 17.3 / 14.8 |
| ur5e_2f85 | stack_overhang | 0.75 (0.81) | 0.62 (0.77) | 0.75 (0.81) | topple 2 | topple 2 | 45.9 / 75.4 / 45.9 |
