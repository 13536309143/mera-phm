# MERA-PHM equipment-grouped validation

Exploratory internal evaluation on an already explored synthetic dataset. Each of the 50 devices is tested once.

| Method | OOF MAE | Device macro MAE | RMSE | Mean interval width | Whole-device coverage |
|---|---:|---:|---:|---:|---:|
| numeric | 6.4884 | 6.4900 | 9.0375 | 71.68 | 94% |
| text | 9.6095 | 9.6123 | 15.9786 | 142.69 | 92% |
| image | 13.4012 | 13.3576 | 17.7589 | 132.13 | 92% |
| numeric_text | 3.9980 | 3.9999 | 7.1479 | 64.17 | 94% |
| numeric_image | 3.4601 | 3.4615 | 5.4967 | 64.65 | 92% |
| static_fusion | 2.9759 | 2.9758 | 4.9346 | 59.55 | 98% |
| history_fusion | 2.7834 | 2.7860 | 4.6642 | 57.15 | 100% |
| event_history | 2.8645 | 2.8675 | 4.8098 | 75.30 | 98% |
| missing_rule | 2.8753 | 2.8755 | 5.0030 | 64.15 | 94% |
| learned_router | 2.6919 | 2.6958 | 4.7726 | 64.15 | 94% |
| router_no_age | 2.6936 | 2.6972 | 4.7752 | 64.15 | 94% |
| router_no_event | 2.6888 | 2.6924 | 4.7755 | 64.15 | 94% |
| router_unconstrained | 2.7891 | 2.7919 | 4.7931 | 67.92 | 96% |
| dummy_mean | 22.7546 | 22.6768 | 26.6570 | 100.90 | 100% |
| ridge_fusion_25 | 8.1288 | 8.1016 | 10.4641 | 86.96 | 96% |
| random_forest_fusion_25 | 3.5489 | 3.5503 | 5.7842 | 65.53 | 96% |
| random_forest_fusion_30 | 3.3124 | 3.3116 | 5.3514 | 53.37 | 90% |
| extra_trees_fusion_30 | 2.7590 | 2.7584 | 4.6994 | 51.33 | 84% |
| hist_gradient_numeric_25 | 6.4699 | 6.4685 | 8.8609 | 63.64 | 94% |
| router_without_freshness | 2.7406 | 2.7427 | 4.8286 | 64.15 | 94% |
| router_without_reset | 2.7384 | 2.7410 | 4.7030 | 58.23 | 96% |

The router fits experts on 25 devices and its gate on 5 more. The 30-device static baselines control for this additional label budget.

## Stress tests

| Condition | Static fusion MAE | Unconstrained router MAE | Constrained router MAE |
|---|---:|---:|---:|
| image_outage | 17.1547 | 14.9869 | 3.9980 |
| text_outage | 5.8902 | 4.9616 | 3.4893 |
| both_outage | 22.7948 | 17.9988 | 6.4884 |
| stale_history | 17.1547 | 15.4095 | 3.9980 |
| random_image_missing_50 | 9.9626 | 8.4083 | 6.2570 |
| block_image_outage | 7.6497 | 7.1600 | 3.4733 |
| curve_feature_noise | 3.4668 | 3.4409 | 3.3473 |

## Interpretation limits

- Same synthetic dataset previously explored; internal validation, not untouched external confirmation.
- Hyperparameters fixed before this run; no original-paper CNN reproduction.
- Bootstrap over devices is descriptive; models share training data across folds.
- Shifted stress tests do not inherit nominal conformal coverage guarantees.

Complete fold assignments, parameters, component definitions, widths, and descriptive paired-device bootstrap estimates are in the adjacent JSON files.
