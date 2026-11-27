
**bin_clearing / gen3_2f85** (heldout_robot, n = 30, env envs-r7.1, best scripted = PTA (probe-then-act))

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.30 | 56.7 | 88.5 | 0.57 | **+0.20 [+0.07,+0.33]** | **+0.30 [+0.13,+0.47]** | – | – |
| PTA (probe-then-act) | 0.10 | 78.7 | 98.2 | 0.10 | – | +0.10 [+0.00,+0.23] | – | – |
| oracle-param scripted (labelled) | 0.37 | 78.7 | 91.7 | 0.00 | **+0.27 [+0.10,+0.43]** | **+0.37 [+0.20,+0.53]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 101.1 | 0.00 | -0.10 [-0.20,+0.00] | – | – | – |
| no-memory, budget-matched | 0.20 | 66.6 | 93.1 | 0.70 | +0.10 [-0.03,+0.23] | **+0.20 [+0.07,+0.37]** | – | – |
| generic memory, budget-matched | 0.20 | 55.4 | 91.2 | 0.70 | +0.10 [-0.03,+0.23] | **+0.20 [+0.07,+0.37]** | – | +0.00 [+0.00,+0.00] |
| typed memory, budget-matched | 0.23 | 46.3 | 88.8 | 0.73 | +0.13 [+0.00,+0.30] | **+0.23 [+0.10,+0.40]** | +0.03 [+0.00,+0.10] | +0.03 [+0.00,+0.10] |
| no-memory, full data | 0.17 | 73.9 | 95.8 | 0.33 | +0.07 [-0.07,+0.20] | **+0.17 [+0.03,+0.30]** | – | – |
| generic memory, full data | 0.23 | 66.4 | 92.9 | 0.40 | **+0.13 [+0.03,+0.27]** | **+0.23 [+0.10,+0.40]** | – | +0.07 [-0.07,+0.20] |
| typed memory, full data | 0.13 | 44.4 | 94.3 | 0.50 | +0.03 [-0.07,+0.13] | **+0.13 [+0.03,+0.27]** | -0.10 [-0.23,+0.00] | -0.03 [-0.10,+0.00] |

**bin_clearing / panda** (test, n = 30, env envs-r7.1, best scripted = PTA (probe-then-act))

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.17 | 47.6 | 104.5 | 0.73 | +0.03 [-0.13,+0.20] | **+0.17 [+0.03,+0.30]** | – | – |
| PTA (probe-then-act) | 0.13 | 57.8 | 107.7 | 0.17 | – | **+0.13 [+0.03,+0.27]** | – | – |
| oracle-param scripted (labelled) | 0.43 | 50.4 | 92.1 | 0.07 | **+0.30 [+0.13,+0.47]** | **+0.43 [+0.27,+0.60]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 114.4 | 0.03 | _-0.13 [-0.27,-0.03]_ | – | – | – |
| no-memory, budget-matched | 0.17 | 55.1 | 104.7 | 0.63 | +0.03 [-0.10,+0.17] | **+0.17 [+0.03,+0.30]** | – | – |
| generic memory, budget-matched | 0.07 | 47.3 | 109.9 | 0.70 | -0.07 [-0.23,+0.07] | +0.07 [+0.00,+0.17] | – | -0.10 [-0.20,+0.00] |
| typed memory, budget-matched | 0.10 | 44.3 | 107.5 | 0.70 | -0.03 [-0.20,+0.13] | +0.10 [+0.00,+0.20] | +0.03 [+0.00,+0.10] | -0.07 [-0.17,+0.00] |
| no-memory, full data | 0.07 | 71.2 | 111.5 | 0.37 | -0.07 [-0.23,+0.10] | +0.07 [+0.00,+0.17] | – | – |
| generic memory, full data | 0.20 | 60.3 | 104.5 | 0.23 | +0.07 [-0.07,+0.20] | **+0.20 [+0.07,+0.37]** | – | **+0.13 [+0.03,+0.27]** |
| typed memory, full data | 0.17 | 61.2 | 106.5 | 0.20 | +0.03 [-0.10,+0.17] | **+0.17 [+0.03,+0.30]** | -0.03 [-0.17,+0.07] | +0.10 [-0.03,+0.23] |

**bin_clearing / ur5e_2f85** (test, n = 30, env envs-r7.1, best scripted = PTA (probe-then-act))

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.20 | 42.9 | 121.5 | 0.73 | -0.03 [-0.17,+0.10] | **+0.20 [+0.07,+0.37]** | – | – |
| PTA (probe-then-act) | 0.23 | 67.2 | 124.0 | 0.00 | – | **+0.23 [+0.10,+0.40]** | – | – |
| oracle-param scripted (labelled) | 0.57 | 73.1 | 101.8 | 0.00 | **+0.33 [+0.13,+0.53]** | **+0.57 [+0.40,+0.73]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 140.5 | 0.00 | _-0.23 [-0.40,-0.10]_ | – | – | – |
| no-memory, budget-matched | 0.20 | 55.9 | 124.3 | 0.73 | -0.03 [-0.23,+0.17] | **+0.20 [+0.07,+0.33]** | – | – |
| generic memory, budget-matched | 0.20 | 66.5 | 126.9 | 0.77 | -0.03 [-0.23,+0.17] | **+0.20 [+0.07,+0.37]** | – | +0.00 [-0.10,+0.10] |
| typed memory, budget-matched | 0.17 | 54.6 | 127.6 | 0.77 | -0.07 [-0.27,+0.10] | **+0.17 [+0.03,+0.30]** | -0.03 [-0.10,+0.00] | -0.03 [-0.13,+0.07] |
| no-memory, full data | 0.10 | 54.0 | 131.4 | 0.63 | _-0.13 [-0.27,-0.03]_ | +0.10 [+0.00,+0.20] | – | – |
| generic memory, full data | 0.33 | 51.2 | 112.9 | 0.40 | +0.10 [-0.03,+0.23] | **+0.33 [+0.17,+0.50]** | – | **+0.23 [+0.10,+0.40]** |
| typed memory, full data | 0.20 | 63.8 | 125.2 | 0.57 | -0.03 [-0.20,+0.13] | **+0.20 [+0.07,+0.33]** | _-0.13 [-0.27,-0.03]_ | +0.10 [-0.07,+0.27] |

**fixture_place / panda** (test, n = 30, env envs-r7.1, best scripted = nominal scripted)

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.67 | 11.8 | 14.8 | 0.17 | – | **+0.60 [+0.40,+0.80]** | – | – |
| PTA (probe-then-act) | 0.43 | 11.8 | 17.5 | 0.03 | -0.23 [-0.47,+0.00] | **+0.37 [+0.17,+0.57]** | – | – |
| oracle-param scripted (labelled) | 0.73 | 11.7 | 14.3 | 0.00 | +0.07 [-0.10,+0.27] | **+0.67 [+0.50,+0.83]** | – | – |
| ACT imitation (tuned on VAL) | 0.07 | 10.2 | 19.3 | 0.10 | _-0.60 [-0.80,-0.40]_ | – | – | – |
| no-memory, budget-matched | 0.37 | 11.9 | 17.4 | 0.43 | _-0.30 [-0.53,-0.07]_ | **+0.30 [+0.07,+0.50]** | – | – |
| generic memory, budget-matched | 0.40 | 10.0 | 16.5 | 0.37 | _-0.27 [-0.47,-0.07]_ | **+0.33 [+0.13,+0.53]** | – | +0.03 [-0.17,+0.23] |
| typed memory, budget-matched | 0.53 | 10.3 | 15.3 | 0.23 | -0.13 [-0.27,+0.03] | **+0.47 [+0.23,+0.67]** | +0.13 [-0.07,+0.33] | +0.17 [-0.07,+0.40] |
| no-memory, full data | 0.57 | 11.7 | 15.3 | 0.17 | -0.10 [-0.23,+0.03] | **+0.50 [+0.27,+0.70]** | – | – |
| generic memory, full data | 0.60 | 11.9 | 15.2 | 0.13 | -0.07 [-0.30,+0.13] | **+0.53 [+0.33,+0.73]** | – | +0.03 [-0.20,+0.23] |
| typed memory, full data | 0.70 | 11.9 | 14.8 | 0.17 | +0.03 [+0.00,+0.10] | **+0.63 [+0.43,+0.80]** | +0.10 [-0.13,+0.30] | +0.13 [+0.00,+0.27] |

**fixture_place / ur5e_2f85** (test, n = 30, env envs-r7.1, best scripted = nominal scripted)

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.67 | 11.7 | 14.7 | 0.07 | – | **+0.60 [+0.40,+0.77]** | – | – |
| PTA (probe-then-act) | 0.57 | 11.7 | 15.8 | 0.07 | -0.10 [-0.30,+0.10] | **+0.50 [+0.33,+0.67]** | – | – |
| oracle-param scripted (labelled) | 0.63 | 11.7 | 14.7 | 0.00 | -0.03 [-0.17,+0.07] | **+0.57 [+0.40,+0.73]** | – | – |
| ACT imitation (tuned on VAL) | 0.07 | 11.0 | 19.4 | 0.13 | _-0.60 [-0.77,-0.43]_ | – | – | – |
| no-memory, budget-matched | 0.30 | 11.2 | 17.4 | 0.47 | _-0.37 [-0.57,-0.17]_ | **+0.23 [+0.03,+0.43]** | – | – |
| generic memory, budget-matched | 0.40 | 9.9 | 16.2 | 0.33 | _-0.27 [-0.47,-0.07]_ | **+0.33 [+0.13,+0.53]** | – | +0.10 [-0.13,+0.33] |
| typed memory, budget-matched | 0.60 | 9.9 | 14.9 | 0.27 | -0.07 [-0.27,+0.13] | **+0.53 [+0.33,+0.73]** | **+0.20 [+0.07,+0.33]** | **+0.30 [+0.07,+0.53]** |
| no-memory, full data | 0.57 | 11.4 | 15.6 | 0.27 | -0.10 [-0.27,+0.07] | **+0.50 [+0.27,+0.70]** | – | – |
| generic memory, full data | 0.57 | 11.4 | 15.3 | 0.23 | -0.10 [-0.27,+0.07] | **+0.50 [+0.30,+0.70]** | – | +0.00 [-0.17,+0.20] |
| typed memory, full data | 0.63 | 11.7 | 15.1 | 0.27 | -0.03 [-0.20,+0.13] | **+0.57 [+0.37,+0.77]** | +0.07 [-0.07,+0.20] | +0.07 [-0.07,+0.20] |

**tight_insert / gen3_2f85** (heldout_robot, n = 30, env envs-r7.1, best scripted = force-ramp)

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.67 | 19.1 | 29.6 | 0.07 | -0.10 [-0.20,+0.00] | **+0.67 [+0.50,+0.83]** | – | – |
| PTA (probe-then-act) | 0.40 | 18.1 | 37.2 | 0.27 | _-0.37 [-0.53,-0.20]_ | **+0.40 [+0.23,+0.57]** | – | – |
| force-ramp | 0.77 | 20.1 | 27.5 | 0.00 | – | **+0.77 [+0.60,+0.90]** | – | – |
| oracle-param scripted (labelled) | 0.67 | 18.4 | 29.0 | 0.03 | -0.10 [-0.20,+0.00] | **+0.67 [+0.50,+0.83]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 48.0 | 0.23 | _-0.77 [-0.90,-0.60]_ | – | – | – |
| no-memory, budget-matched | 0.40 | 22.6 | 38.1 | 0.07 | _-0.37 [-0.57,-0.13]_ | **+0.40 [+0.23,+0.57]** | – | – |
| generic memory, budget-matched | 0.43 | 23.7 | 37.3 | 0.10 | _-0.33 [-0.53,-0.10]_ | **+0.43 [+0.27,+0.60]** | – | +0.03 [-0.10,+0.17] |
| typed memory, budget-matched | 0.43 | 21.2 | 36.9 | 0.07 | _-0.33 [-0.50,-0.17]_ | **+0.43 [+0.27,+0.60]** | +0.00 [-0.13,+0.13] | +0.03 [-0.17,+0.23] |
| no-memory, full data | 0.73 | 19.0 | 27.0 | 0.13 | -0.03 [-0.23,+0.17] | **+0.73 [+0.57,+0.90]** | – | – |
| generic memory, full data | 0.77 | 18.9 | 25.9 | 0.13 | +0.00 [-0.20,+0.20] | **+0.77 [+0.60,+0.90]** | – | +0.03 [-0.07,+0.13] |
| typed memory, full data | 0.67 | 19.4 | 29.4 | 0.10 | -0.10 [-0.23,+0.03] | **+0.67 [+0.50,+0.83]** | -0.10 [-0.27,+0.07] | -0.07 [-0.23,+0.10] |

**tight_insert / panda** (test, n = 30, env envs-r7.1, best scripted = force-ramp)

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.37 | 15.9 | 24.1 | 0.27 | -0.10 [-0.20,+0.00] | **+0.37 [+0.20,+0.53]** | – | – |
| PTA (probe-then-act) | 0.30 | 19.2 | 25.4 | 0.17 | -0.17 [-0.33,+0.00] | **+0.30 [+0.13,+0.47]** | – | – |
| force-ramp | 0.47 | 18.8 | 23.9 | 0.23 | – | **+0.47 [+0.30,+0.63]** | – | – |
| oracle-param scripted (labelled) | 0.43 | 15.7 | 22.9 | 0.27 | -0.03 [-0.13,+0.07] | **+0.43 [+0.27,+0.60]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 28.0 | 0.27 | _-0.47 [-0.63,-0.30]_ | – | – | – |
| no-memory, budget-matched | 0.43 | 20.5 | 24.6 | 0.17 | -0.03 [-0.17,+0.10] | **+0.43 [+0.27,+0.60]** | – | – |
| generic memory, budget-matched | 0.47 | 19.2 | 23.8 | 0.23 | +0.00 [-0.20,+0.20] | **+0.47 [+0.30,+0.63]** | – | +0.03 [-0.13,+0.20] |
| typed memory, budget-matched | 0.50 | 19.9 | 23.9 | 0.10 | +0.03 [-0.13,+0.20] | **+0.50 [+0.33,+0.70]** | +0.03 [-0.10,+0.17] | +0.07 [-0.07,+0.20] |
| no-memory, full data | 0.47 | 17.9 | 23.5 | 0.17 | +0.00 [-0.13,+0.13] | **+0.47 [+0.30,+0.63]** | – | – |
| generic memory, full data | 0.47 | 17.9 | 23.9 | 0.23 | +0.00 [-0.20,+0.20] | **+0.47 [+0.30,+0.63]** | – | +0.00 [-0.17,+0.20] |
| typed memory, full data | 0.47 | 18.5 | 23.6 | 0.17 | +0.00 [+0.00,+0.00] | **+0.47 [+0.30,+0.63]** | +0.00 [-0.20,+0.20] | +0.00 [-0.13,+0.13] |

**tight_insert / ur5e_2f85** (test, n = 30, env envs-r7.1, best scripted = force-ramp)

| policy | success (in budget) | median cycle s | mean cycle s (fail = cap) | damage | Δ vs best scripted | Δ vs ACT | Δ vs generic (same regime) | Δ vs no-memory (same regime) |
|---|---|---|---|---|---|---|---|---|
| nominal scripted | 0.57 | 16.7 | 22.9 | 0.20 | _-0.13 [-0.27,-0.03]_ | **+0.57 [+0.37,+0.73]** | – | – |
| PTA (probe-then-act) | 0.60 | 22.2 | 25.1 | 0.17 | -0.10 [-0.33,+0.13] | **+0.60 [+0.43,+0.77]** | – | – |
| force-ramp | 0.70 | 19.1 | 22.4 | 0.10 | – | **+0.70 [+0.53,+0.87]** | – | – |
| oracle-param scripted (labelled) | 0.67 | 16.4 | 21.6 | 0.13 | -0.03 [-0.17,+0.07] | **+0.67 [+0.50,+0.83]** | – | – |
| ACT imitation (tuned on VAL) | 0.00 | – | 30.0 | 0.03 | _-0.70 [-0.87,-0.53]_ | – | – | – |
| no-memory, budget-matched | 0.43 | 19.1 | 26.1 | 0.13 | _-0.27 [-0.50,-0.03]_ | **+0.43 [+0.27,+0.60]** | – | – |
| generic memory, budget-matched | 0.30 | 22.5 | 27.9 | 0.10 | _-0.40 [-0.63,-0.17]_ | **+0.30 [+0.13,+0.47]** | – | -0.13 [-0.30,+0.03] |
| typed memory, budget-matched | 0.40 | 22.2 | 27.2 | 0.10 | _-0.30 [-0.53,-0.07]_ | **+0.40 [+0.23,+0.57]** | +0.10 [+0.00,+0.20] | -0.03 [-0.17,+0.10] |
| no-memory, full data | 0.60 | 16.5 | 22.5 | 0.13 | -0.10 [-0.30,+0.10] | **+0.60 [+0.43,+0.77]** | – | – |
| generic memory, full data | 0.63 | 17.2 | 22.9 | 0.10 | -0.07 [-0.27,+0.13] | **+0.63 [+0.47,+0.80]** | – | +0.03 [-0.17,+0.23] |
| typed memory, full data | 0.50 | 17.1 | 23.8 | 0.10 | _-0.20 [-0.37,-0.07]_ | **+0.50 [+0.33,+0.67]** | -0.13 [-0.33,+0.07] | -0.10 [-0.27,+0.07] |
