success (mean progress) over seeds; site-dressed; tag r6

| robot | family | nominal | oracle | oracle_slow | oracle_firm | oracle failure causes | nominal failure causes | median success sim-s (nominal / oracle / oracle_slow / oracle_firm) |
|---|---|---|---|---|---|---|---|---|
| gen3_2f85 | bin_clearing | 0.00 (0.62) | 0.17 (0.45) | 0.17 (0.49) | 0.17 (0.51) | grasp_miss 4, crush_or_damage 1 | crush_or_damage 5, grasp_miss 1 | - / 35.7 / 45.9 / 35.6 |
| gen3_2f85 | conveyor_sort | 0.00 (0.50) | 0.00 (0.30) | 0.00 (0.20) | 0.00 (0.27) | grasp_miss 2, ik_fail 2, crush_or_damage 2 | crush_or_damage 4, grasp_miss 1, ik_fail 1 | - / - / - / - |
| gen3_2f85 | depalletize | 0.00 (0.00) | 0.00 (0.56) | 0.00 (0.56) | 0.00 (0.56) | grasp_miss 6 | grasp_miss 6 | - / - / - / - |
| gen3_2f85 | fixture_place | 0.33 (0.83) | 0.50 (0.83) | 0.50 (0.76) | 0.50 (0.96) | crush_or_damage 2, slip 1 | crush_or_damage 3, slip 1 | 11.5 / 11.6 / 17.4 / 11.5 |
| gen3_2f85 | pouch_transfer | 0.17 (0.33) | 0.33 (0.33) | 0.17 (0.17) | 0.33 (0.33) | slip 4 | slip 4, crush_or_damage 1 | 11.6 / 11.6 / 15.8 / 11.5 |
| gen3_2f85 | tool_return | 0.50 (0.50) | 0.50 (0.50) | 0.50 (0.50) | 0.33 (0.44) | slip 2, grasp_miss 1 | grasp_miss 2, slip 1 | 49.2 / 22.6 / 56.7 / 37.6 |
| panda | bin_clearing | 0.00 (0.32) | 0.00 (0.29) | 0.00 (0.17) | 0.00 (0.22) | grasp_miss 6 | grasp_miss 4, crush_or_damage 2 | - / - / - / - |
| panda | conveyor_sort | 0.17 (0.62) | 0.00 (0.29) | 0.00 (0.22) | 0.00 (0.39) | grasp_miss 3, crush_or_damage 2, ik_fail 1 | crush_or_damage 4, grasp_miss 1 | 65.3 / - / - / - |
| panda | depalletize | 0.00 (0.00) | 0.50 (0.75) | 0.50 (0.75) | 0.50 (0.75) | grasp_miss 3 | grasp_miss 6 | - / 8.1 / 8.1 / 8.1 |
| panda | fixture_place | 1.00 (1.00) | 0.83 (1.00) | 1.00 (1.00) | 1.00 (1.00) | crush_or_damage 1 |  | 11.6 / 10.7 / 18.5 / 11.6 |
| panda | pouch_transfer | 0.33 (0.33) | 0.33 (0.33) | 0.33 (0.33) | 0.33 (0.33) | grasp_miss 2, crush_or_damage 1, slip 1 | crush_or_damage 4 | 11.4 / 11.4 / 15.6 / 11.4 |
| panda | tool_return | 0.17 (0.33) | 0.33 (0.50) | 0.33 (0.50) | 0.17 (0.17) | crush_or_damage 3, jam 1 | crush_or_damage 4, grasp_miss 1 | 43.2 / 30.7 / 39.2 / 14.7 |
| probe | bin_clearing | 0.17 (0.97) | 0.33 (0.93) | 0.67 (0.96) | 0.67 (1.00) | crush_or_damage 3, slip 1 | crush_or_damage 5 | 61.0 / 46.3 / 75.6 / 51.8 |
| probe | conveyor_sort | 0.33 (0.86) | 0.67 (0.82) | 0.00 (0.17) | 0.50 (0.79) | grasp_miss 2 | crush_or_damage 4 | 59.0 / 66.3 / - / 65.9 |
| probe | depalletize | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |  |  | 99.0 / 95.1 / 141.1 / 95.1 |
| probe | fixture_place | 0.83 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |  | crush_or_damage 1 | 12.9 / 12.8 / 22.4 / 12.8 |
| probe | pouch_transfer | 0.00 (1.00) | 0.33 (0.33) | 0.00 (0.17) | 0.00 (0.67) | slip 4 | crush_or_damage 6 | - / 15.6 / - / - |
| probe | tool_return | 1.00 (1.00) | 1.00 (1.00) | 0.83 (0.83) | 1.00 (1.00) |  |  | 17.6 / 17.6 / 26.7 / 17.6 |
| ur5e_2f85 | bin_clearing | 0.00 (0.70) | 0.17 (0.48) | 0.17 (0.31) | 0.00 (0.60) | grasp_miss 4, slip 1 | crush_or_damage 6 | - / 64.8 / 85.2 / - |
| ur5e_2f85 | conveyor_sort | 0.17 (0.72) | 0.17 (0.35) | 0.00 (0.32) | 0.17 (0.51) | crush_or_damage 4, grasp_miss 1 | crush_or_damage 4, collision 1 | 65.7 / 53.7 / - / 53.5 |
| ur5e_2f85 | depalletize | 0.00 (0.00) | 0.17 (0.35) | 0.17 (0.35) | 0.17 (0.35) | grasp_miss 5 | grasp_miss 6 | - / 8.1 / 8.1 / 8.1 |
| ur5e_2f85 | fixture_place | 0.83 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |  | crush_or_damage 1 | 11.6 / 11.6 / 18.5 / 11.6 |
| ur5e_2f85 | pouch_transfer | 0.17 (0.33) | 0.33 (0.33) | 0.33 (0.33) | 0.17 (0.33) | grasp_miss 2, slip 1, crush_or_damage 1 | crush_or_damage 5 | 10.9 / 11.2 / 15.6 / 10.9 |
| ur5e_2f85 | tool_return | 0.67 (0.67) | 0.67 (0.67) | 0.67 (0.67) | 0.67 (0.67) | grasp_miss 1, crush_or_damage 1 | crush_or_damage 1, grasp_miss 1 | 22.8 / 22.9 / 25.5 / 22.8 |
