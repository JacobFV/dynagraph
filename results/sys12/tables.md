## Offline System I outcome prediction (fixed row subsets, <=512 rows per family, row-weighted pool)

| model | split | rows | success_auc | success_brier | damage_auc | slip_auc | drop_auc | cont_nll | effort_log_rmse |
|---|---|---|---|---|---|---|---|---|---|
| Inkling-2-Mini core (pretrained), gated System I, 1 pass | test | 7369 | 0.844 | 0.050 | 0.730 | 0.879 | 0.562 | -1.146 | 2.888 |
| Inkling-2-Mini core (pretrained), gated System I, 1 pass | heldout_robot | 0 | – | – | – | – | – | – | – |
| Inkling-2-Mini core (pretrained), gated System I, 1 pass | heldout_family | 1536 | 0.833 | 0.091 | 0.506 | 0.706 | 0.784 | 7.930 | 3.847 |
| Inkling-2-Mini core (pretrained), UNGATED LoRA (control i) | test | 7369 | 0.847 | 0.048 | 0.702 | 0.886 | 0.510 | -0.979 | 3.432 |
| Inkling-2-Mini core (pretrained), UNGATED LoRA (control i) | heldout_robot | 0 | – | – | – | – | – | – | – |
| Inkling-2-Mini core (pretrained), UNGATED LoRA (control i) | heldout_family | 1536 | 0.841 | 0.093 | 0.539 | 0.735 | 0.785 | 1.672 | 3.817 |
| Inkling random-init core, gated (control ii) | test | 7369 | 0.840 | 0.050 | 0.717 | 0.862 | 0.631 | -0.799 | 3.244 |
| Inkling random-init core, gated (control ii) | heldout_robot | 0 | – | – | – | – | – | – | – |
| Inkling random-init core, gated (control ii) | heldout_family | 1536 | 0.846 | 0.090 | 0.535 | 0.738 | 0.807 | 3.407 | 4.036 |
| compact System I, same frozen memory (control iii-a) | test | 7369 | 0.826 | 0.049 | 0.710 | 0.885 | 0.538 | -1.350 | 2.767 |
| compact System I, same frozen memory (control iii-a) | heldout_robot | 0 | – | – | – | – | – | – | – |
| compact System I, same frozen memory (control iii-a) | heldout_family | 1536 | 0.795 | 0.095 | 0.524 | 0.794 | 0.785 | 8.298 | 4.155 |
| MODEL-V2 typed, sample-matched 128k (control iii-b) | test | 7369 | 0.753 | 0.049 | 0.411 | 0.280 | 0.476 | -0.430 | 3.667 |
| MODEL-V2 typed, sample-matched 128k (control iii-b) | heldout_robot | 0 | – | – | – | – | – | – | – |
| MODEL-V2 typed, sample-matched 128k (control iii-b) | heldout_family | 1536 | 0.640 | 0.101 | 0.347 | 0.305 | 0.596 | 0.209 | 4.570 |
| MODEL-V2 r7 typed s0 (2.05 M samples, reference) | test | 7369 | 0.869 | 0.040 | 0.816 | 0.910 | 0.634 | -1.643 | 2.685 |
| MODEL-V2 r7 typed s0 (2.05 M samples, reference) | heldout_robot | 0 | – | – | – | – | – | – | – |
| MODEL-V2 r7 typed s0 (2.05 M samples, reference) | heldout_family | 1536 | 0.867 | 0.097 | 0.640 | 0.799 | 0.692 | 285.320 | 4.370 |

| run | samples seen | train s | trainable params |
|---|---|---|---|
| Inkling-2-Mini core (pretrained), gated System I, 1 pass | 128032 | 12350.1 | 5541033 |
| Inkling-2-Mini core (pretrained), UNGATED LoRA (control i) | 128032 | 8611.6 | 5541033 |
| Inkling random-init core, gated (control ii) | 128032 | 16360.8 | 5541033 |
| compact System I, same frozen memory (control iii-a) | 128032 | 93.3 | 1292195 |

## Gate test (gate_test.json; adapters: trained adapters from results/sys12/checkpoints/smoke_gated.pt)

| check | value |
|---|---|
| a_instrumented_equals_native | True |
| b_system2_logits_bit_identical | True |
| b_system2_max_abs_logit_diff | 0.0 |
| b_system2_greedy_identical | True |
| c_ungated_changes_logits | True |
| c_ungated_max_abs_logit_diff | 4.074775695800781 |
| c_ungated_mean_KL_base_to_ungated | 0.004790156148374081 |
| d_sys1_prefix_states_bit_identical_p1 | False |
| d_sys1_prefix_max_abs_diff_p1 | 6.5 |
| d_sys1_prefix_states_bit_identical_p3 | False |
| d_sys1_prefix_max_abs_diff_p3 | 6.96875 |
| latency sys1_bs1_passes1_fast_ms | 305.1 |
| latency sys1_bs1_passes2_fast_ms | 497.8 |
| latency sys1_bs1_passes3_fast_ms | 671.4 |
| latency sys1_bs1_passes1_exact_ms | 410.8 |
| latency sys1_bs1_passes2_exact_ms | 651.1 |
| latency sys1_bs1_passes3_exact_ms | 914.8 |
| latency sys2_menu_forward_bs3_ms | 774.3 |

## Gate test (gate_test_trained.json; adapters: trained adapters from results/sys12/checkpoints/s1r1_inkling_gated_s0.pt)

| check | value |
|---|---|
| a_instrumented_equals_native | True |
| b_system2_logits_bit_identical | True |
| b_system2_max_abs_logit_diff | 0.0 |
| b_system2_greedy_identical | True |
| c_ungated_changes_logits | True |
| c_ungated_max_abs_logit_diff | 32.10914993286133 |
| c_ungated_mean_KL_base_to_ungated | 0.9885401030381521 |
| d_sys1_prefix_states_bit_identical_to_base_same_seq_p1 | False |
| d_sys1_prefix_max_abs_diff_same_seq_p1 | 5.0 |
| d_sys1_readout_changed_by_adapters_p1 | True |
| d_base_full_vs_prefix_only_rel_diff_p1 | 0.029329191893339157 |
| d_sys1_prefix_states_bit_identical_to_base_same_seq_p3 | False |
| d_sys1_prefix_max_abs_diff_same_seq_p3 | 10.0 |
| d_sys1_readout_changed_by_adapters_p3 | True |
| d_base_full_vs_prefix_only_rel_diff_p3 | 0.060432933270931244 |
| latency sys1_bs1_passes1_fast_ms | 607.7 |
| latency sys1_bs1_passes2_fast_ms | 1013.7 |
| latency sys1_bs1_passes3_fast_ms | 1341.9 |
| latency sys1_bs1_passes1_exact_ms | 1844.0 |
| latency sys1_bs1_passes2_exact_ms | 2917.7 |
| latency sys1_bs1_passes3_exact_ms | 4075.2 |
| latency sys2_menu_forward_bs3_ms | 3390.6 |

## Language retention (Salesforce/wikitext wikitext-2-raw-v1 test (public), 24576 tokens)

| condition | ppl | KL(base‖cond) top-64 | parse format | parse skill | parse target | parse exact | parse constrained | instr acc |
|---|---|---|---|---|---|---|---|---|
| base | 8.337 | 0.00000 | 0.03 | 0.03 | 0.03 | 0.03 | 0.83 | 0.06 (16) |
| gated_system2 | 8.337 | 0.00000 | 0.03 | 0.03 | 0.03 | 0.03 | 0.83 | 0.06 (16) |
| ungated | 52.805 | 1.38466 | 0.00 | 0.00 | 0.00 | 0.00 | 0.45 | 0.00 (16) |

## Closed loop bin_gen3 (bin_clearing, gen3_2f85, heldout_robot, n = 30; primary = good parts per lot)

| policy | primary | success | damage | probes/ep | effort | steps | timeouts |
|---|---|---|---|---|---|---|---|
| heuristic | 0.12 | 0.00 | 0.50 | 0.00 | 60674 | 6.9 | 3 |
| pta | 0.22 | 0.07 | 0.20 | 2.47 | 60927 | 6.3 | 3 |
| v2 | 0.22 | 0.00 | 0.07 | 1.20 | 60795 | 7.2 | 1 |
| s12_native | 0.21 | 0.07 | 0.43 | 0.00 | 69850 | 7.1 | 2 |
| s1native_scripted | 0.20 | 0.03 | 0.53 | 0.00 | 77499 | 6.9 | 2 |
| s12_delegated | 0.30 | 0.03 | 0.37 | 0.17 | 25249 | 7.6 | 1 |

| pre-registered comparison | Δ primary [95% CI] | Δ damage [95% CI] |
|---|---|---|
| C1 s12_native − v2 | +0.000 [-0.119, +0.131] | +0.37 [+0.20, +0.53] |
| C2 s12_native − pta | +0.000 [-0.099, +0.087] | +0.23 [+0.03, +0.43] |
| C3 s12_delegated − v2 | +0.080 [-0.023, +0.184] | +0.30 [+0.13, +0.47] |
| C4 s12_delegated − pta | +0.074 [-0.012, +0.173] | +0.17 [-0.07, +0.40] |
| C5 s12_native − s12_delegated | **-0.095 [-0.190, -0.012]** | +0.07 [-0.10, +0.23] |
| C6 s12_native − s1native_scripted | +0.012 [+0.000, +0.036] | -0.10 [-0.20, +0.00] |

| policy | decisions | System II agrees with scripted | forced (1 option) | sys2 ms (median, LM calls) | sys1 ms (median) | probe decisions |
|---|---|---|---|---|---|---|
| s12_delegated | 230 | 0.74 | 0.38 | 36251 | 1818 | 5 |
| s12_native | 217 | 0.77 | 0.40 | 33304 | 1765 | 0 |
| s1native_scripted | 213 | 1.00 | 0.49 | – | 1748 | 0 |

## Closed loop bin_ur5e (bin_clearing, ur5e_2f85, test, n = 30; primary = good parts per lot)

| policy | primary | success | damage | probes/ep | effort | steps | timeouts |
|---|---|---|---|---|---|---|---|
| heuristic | 0.16 | 0.03 | 0.53 | 0.00 | 39058 | 8.5 | 1 |
| pta | 0.30 | 0.07 | 0.23 | 3.13 | 186733 | 7.8 | 2 |
| v2 | 0.21 | 0.03 | 0.13 | 1.50 | 171422 | 7.9 | 2 |
| s12_native | 0.18 | 0.03 | 0.33 | 0.00 | 11013 | 8.3 | 2 |
| s1native_scripted | 0.20 | 0.03 | 0.37 | 0.00 | 17772 | 7.5 | 2 |
| s12_delegated | 0.23 | 0.03 | 0.27 | 0.60 | 5151130336120 | 8.2 | 1 |

| pre-registered comparison | Δ primary [95% CI] | Δ damage [95% CI] |
|---|---|---|
| C1 s12_native − v2 | -0.026 [-0.141, +0.103] | +0.20 [+0.07, +0.37] |
| C2 s12_native − pta | -0.128 [-0.269, +0.013] | +0.10 [-0.07, +0.27] |
| C3 s12_delegated − v2 | +0.012 [-0.074, +0.099] | +0.13 [+0.03, +0.27] |
| C4 s12_delegated − pta | -0.074 [-0.210, +0.049] | +0.03 [-0.13, +0.20] |
| C5 s12_native − s12_delegated | -0.048 [-0.143, +0.060] | +0.07 [-0.07, +0.20] |
| C6 s12_native − s1native_scripted | -0.024 [-0.071, +0.024] | -0.03 [-0.10, +0.00] |

| policy | decisions | System II agrees with scripted | forced (1 option) | sys2 ms (median, LM calls) | sys1 ms (median) | probe decisions |
|---|---|---|---|---|---|---|
| s12_delegated | 252 | 0.71 | 0.32 | 8478 | 929 | 19 |
| s12_native | 258 | 0.65 | 0.29 | 8755 | 651 | 0 |
| s1native_scripted | 234 | 1.00 | 0.40 | – | 695 | 0 |

## Closed loop pouch_ur5e (pouch_transfer, ur5e_2f85, test, n = 30; primary = success)

| policy | primary | success | damage | probes/ep | effort | steps | timeouts |
|---|---|---|---|---|---|---|---|
| heuristic | 0.20 | 0.20 | 0.67 | 0.00 | 3851 | 4.0 | 0 |
| pta | 0.57 | 0.57 | 0.07 | 3.03 | 3322 | 3.8 | 0 |
| v2 | 0.57 | 0.57 | 0.07 | 0.00 | 3311 | 3.8 | 0 |
| s12_native | 0.30 | 0.30 | 0.03 | 0.00 | 3294 | 4.8 | 0 |
| s1native_scripted | 0.30 | 0.30 | 0.03 | 0.00 | 3294 | 4.8 | 0 |
| s12_delegated | 0.40 | 0.40 | 0.03 | 0.20 | 3261 | 4.3 | 0 |

| pre-registered comparison | Δ primary [95% CI] | Δ damage [95% CI] |
|---|---|---|
| C1 s12_native − v2 | **-0.267 [-0.433, -0.133]** | -0.03 [-0.13, +0.07] |
| C2 s12_native − pta | **-0.267 [-0.433, -0.133]** | -0.03 [-0.13, +0.07] |
| C3 s12_delegated − v2 | **-0.167 [-0.300, -0.033]** | -0.03 [-0.17, +0.07] |
| C4 s12_delegated − pta | **-0.167 [-0.300, -0.033]** | -0.03 [-0.17, +0.07] |
| C5 s12_native − s12_delegated | -0.100 [-0.267, +0.067] | +0.00 [-0.10, +0.10] |
| C6 s12_native − s1native_scripted | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] |

| policy | decisions | System II agrees with scripted | forced (1 option) | sys2 ms (median, LM calls) | sys1 ms (median) | probe decisions |
|---|---|---|---|---|---|---|
| s12_delegated | 128 | 1.00 | 1.00 | – | 2129 | 6 |
| s12_native | 144 | 1.00 | 1.00 | – | 1289 | 0 |
| s1native_scripted | 144 | 1.00 | 1.00 | – | 1295 | 0 |

