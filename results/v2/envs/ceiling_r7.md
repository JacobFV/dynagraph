# envs-r7 ceiling audit (after skill-layer fixes)

Scripted policy per (robot x family), site-dressed, 6 seeds (tight_insert 10); success (mean progress). nominal = fixed engineer params; probe_then_act = fragility-aware scripted (light squeeze -> crush-bounded grip); force_ramp = tight_insert press-until-click (4 N +2 N per stall, 14 N ceiling); oracle = true-hidden-physics params (envs/oracle.py). `@deep_bin` = r6 bin geometry (10 cm walls). Before-fix (envs-r6) table: ceiling_r6.md.

success (mean progress) over seeds; site-dressed; tag r7

| robot | family | nominal | probe_then_act | force_ramp | oracle | oracle failure causes | nominal failure causes | median success sim-s (nominal / probe_then_act / force_ramp / oracle) |
|---|---|---|---|---|---|---|---|---|
| gen3_2f85 | bin_clearing | 0.00 (0.94) | 0.00 (0.61) | - | 0.67 (0.90) | slip 1, grasp_miss 1 | crush_or_damage 6 | - / - / - / 68.2 |
| gen3_2f85 | bin_clearing@deep_bin | 0.00 (0.76) | 0.00 (0.56) | - | 0.33 (0.77) | slip 1, crush_or_damage 1, grasp_miss 1, collision 1 | crush_or_damage 6 | - / - / - / 105.5 |
| gen3_2f85 | conveyor_sort | 0.00 (0.40) | 0.00 (0.30) | - | 0.00 (0.20) | grasp_miss 4, ik_fail 1, crush_or_damage 1 | crush_or_damage 4, grasp_miss 1, ik_fail 1 | - / - / - / - |
| gen3_2f85 | depalletize | 0.00 (0.00) | 0.00 (0.00) | - | 0.00 (0.56) | grasp_miss 6 | grasp_miss 6 | - / - / - / - |
| gen3_2f85 | fastener_pick | 0.00 (0.28) | 0.17 (0.50) | - | 0.00 (0.36) | grasp_miss 5, slip 1 | grasp_miss 4, slip 2 | - / 68.7 / - / - |
| gen3_2f85 | fixture_place | 0.33 (0.66) | 0.50 (0.67) | - | 0.67 (0.67) | slip 2 | crush_or_damage 3, slip 1 | 11.8 / 19.1 / - / 11.8 |
| gen3_2f85 | pouch_transfer | 0.17 (0.17) | 0.00 (0.00) | - | 0.17 (0.17) | slip 3, crush_or_damage 2 | slip 5 | 11.7 / - / - / 11.7 |
| gen3_2f85 | tight_insert | 0.30 (0.65) | - | 0.40 (0.70) | 0.10 (0.62) | jam 4, collision 3, crush_or_damage 2 | jam 3, collision 3, crush_or_damage 1 | 31.7 / - / 28.3 / 32.2 |
| gen3_2f85 | tool_return | 0.17 (0.33) | 0.00 (0.00) | - | 0.00 (0.00) | grasp_miss 5, slip 1 | grasp_miss 4, crush_or_damage 1 | 140.8 / - / - / - |
| panda | bin_clearing | 0.00 (0.77) | 0.33 (0.64) | - | 0.33 (0.75) | grasp_miss 3, slip 1 | crush_or_damage 5, grasp_miss 1 | - / 152.0 / - / 73.0 |
| panda | bin_clearing@deep_bin | 0.00 (0.55) | 0.00 (0.35) | - | 0.00 (0.44) | grasp_miss 6 | crush_or_damage 4, grasp_miss 2 | - / - / - / - |
| panda | conveyor_sort | 0.00 (0.51) | 0.00 (0.39) | - | 0.00 (0.37) | crush_or_damage 3, grasp_miss 2, ik_fail 1 | crush_or_damage 4, place_miss_or_predicate 1, grasp_miss 1 | - / - / - / - |
| panda | depalletize | 0.00 (0.00) | 0.00 (0.00) | - | 0.50 (0.75) | grasp_miss 3 | grasp_miss 6 | - / - / - / 8.1 |
| panda | fastener_pick | 0.00 (0.22) | 0.17 (0.47) | - | 0.00 (0.22) | grasp_miss 5, sim_instability 1 | grasp_miss 5, sim_instability 1 | - / 43.6 / - / - |
| panda | fixture_place | 0.67 (1.00) | 0.67 (0.93) | - | 1.00 (1.00) |  | crush_or_damage 2 | 11.8 / 26.2 / - / 11.9 |
| panda | pouch_transfer | 0.33 (0.33) | 0.33 (0.33) | - | 0.33 (0.33) | crush_or_damage 2, slip 1, grasp_miss 1 | crush_or_damage 4 | 11.5 / 11.6 / - / 11.5 |
| panda | tight_insert | 0.30 (0.67) | - | 0.70 (1.00) | 0.60 (0.90) | crush_or_damage 4 | crush_or_damage 5, jam 2 | 17.1 / - / 19.1 / 14.7 |
| panda | tool_return | 0.17 (0.50) | 0.17 (0.33) | - | 0.17 (0.33) | crush_or_damage 4, grasp_miss 1 | crush_or_damage 4, grasp_miss 1 | 55.5 / 92.3 / - / 72.7 |
| probe | bin_clearing | 0.17 (1.00) | 0.50 (0.92) | - | 0.67 (0.96) | crush_or_damage 1, slip 1 | crush_or_damage 5 | 63.0 / 70.8 / - / 52.3 |
| probe | bin_clearing@deep_bin | 0.17 (1.00) | 0.33 (0.87) | - | 0.67 (0.96) | crush_or_damage 1, slip 1 | crush_or_damage 5 | 63.0 / 76.9 / - / 52.3 |
| probe | conveyor_sort | 0.33 (0.86) | 0.50 (0.79) | - | 0.50 (0.75) | grasp_miss 2, place_miss_or_predicate 1 | crush_or_damage 4 | 59.0 / 65.3 / - / 65.9 |
| probe | depalletize | 1.00 (1.00) | 1.00 (1.00) | - | 1.00 (1.00) |  |  | 99.0 / 99.0 / - / 95.1 |
| probe | fastener_pick | 0.17 (0.44) | 0.00 (0.36) | - | 0.17 (0.44) | slip 4, grasp_miss 1 | slip 3, grasp_miss 2 | 70.9 / - / - / 63.1 |
| probe | fixture_place | 0.83 (1.00) | 1.00 (1.00) | - | 1.00 (1.00) |  | crush_or_damage 1 | 12.9 / 16.4 / - / 12.8 |
| probe | pouch_transfer | 0.00 (1.00) | 0.17 (0.17) | - | 0.33 (0.33) | slip 4 | crush_or_damage 6 | - / 17.1 / - / 15.6 |
| probe | tool_return | 1.00 (1.00) | 1.00 (1.00) | - | 1.00 (1.00) |  |  | 17.6 / 22.4 / - / 17.6 |
| ur5e_2f85 | bin_clearing | 0.17 (0.94) | 0.17 (0.82) | - | 0.83 (0.92) | slip 1 | crush_or_damage 4, grasp_miss 1 | 94.5 / 96.6 / - / 93.6 |
| ur5e_2f85 | bin_clearing@deep_bin | 0.00 (0.88) | 0.17 (0.77) | - | 0.67 (0.90) | slip 2 | crush_or_damage 5, grasp_miss 1 | - / 111.0 / - / 85.6 |
| ur5e_2f85 | conveyor_sort | 0.00 (0.67) | 0.00 (0.36) | - | 0.00 (0.51) | crush_or_damage 4, ik_fail 1, collision 1 | crush_or_damage 5, collision 1 | - / - / - / - |
| ur5e_2f85 | depalletize | 0.00 (0.00) | 0.00 (0.00) | - | 0.17 (0.35) | grasp_miss 5 | grasp_miss 6 | - / - / - / 8.1 |
| ur5e_2f85 | fastener_pick | 0.17 (0.44) | 0.00 (0.44) | - | 0.17 (0.53) | grasp_miss 5 | grasp_miss 5 | 50.8 / - / - / 50.8 |
| ur5e_2f85 | fixture_place | 0.83 (1.00) | 1.00 (1.00) | - | 1.00 (1.00) |  | crush_or_damage 1 | 12.0 / 11.8 / - / 12.1 |
| ur5e_2f85 | pouch_transfer | 0.33 (0.33) | 0.00 (0.00) | - | 0.33 (0.33) | crush_or_damage 3, slip 1 | crush_or_damage 4 | 11.4 / - / - / 11.4 |
| ur5e_2f85 | tight_insert | 0.80 (0.90) | - | 0.90 (1.00) | 0.70 (1.00) | crush_or_damage 1, place_miss_or_predicate 1, grasp_miss 1 | crush_or_damage 1, jam 1 | 17.8 / - / 20.0 / 18.0 |
| ur5e_2f85 | tool_return | 0.00 (0.17) | 0.17 (0.33) | - | 0.50 (0.67) | crush_or_damage 2, slip 1 | crush_or_damage 4, jam 1, grasp_miss 1 | - / 117.3 / - / 204.6 |
