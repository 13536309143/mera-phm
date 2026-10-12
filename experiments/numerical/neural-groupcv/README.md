# MERA-PHM equipment-grouped validation

Exploratory internal evaluation on an already explored synthetic dataset. Each of the 50 devices is tested once.

| Method | OOF MAE | Device macro MAE | RMSE | Mean interval width | Whole-device coverage |
|---|---:|---:|---:|---:|---:|
| numeric | 7.7022 | 7.6942 | 10.1613 | 67.18 | 94% |
| text | 10.4869 | 10.4933 | 17.7238 | 143.61 | 80% |
| image | 15.2621 | 15.2069 | 19.2129 | 129.07 | 92% |
| numeric_text | 7.7799 | 7.7904 | 12.7942 | 100.67 | 88% |
| numeric_image | 4.5756 | 4.5729 | 6.5887 | 60.92 | 92% |
| static_fusion | 7.1451 | 7.1482 | 11.9339 | 111.51 | 90% |
| history_fusion | 6.5719 | 6.5781 | 11.0983 | 106.63 | 88% |
| event_history | 6.5395 | 6.5457 | 11.0170 | 102.56 | 88% |
| missing_rule | 7.1217 | 7.1252 | 11.9013 | 107.15 | 86% |
| learned_router | 6.6297 | 6.6395 | 11.3014 | 110.71 | 90% |
| router_no_age | 6.6576 | 6.6680 | 11.3264 | 110.71 | 90% |
| router_no_event | 6.6320 | 6.6425 | 11.3078 | 110.71 | 90% |
| router_unconstrained | 6.6147 | 6.6241 | 11.3017 | 108.99 | 90% |
| dummy_mean | 22.7546 | 22.6768 | 26.6570 | 100.90 | 100% |
| ridge_fusion_25 | 8.1288 | 8.1016 | 10.4641 | 86.96 | 96% |
| random_forest_fusion_25 | 3.5489 | 3.5503 | 5.7842 | 65.53 | 96% |
| random_forest_fusion_30 | 3.3124 | 3.3116 | 5.3514 | 53.37 | 88% |
| extra_trees_fusion_30 | 2.7590 | 2.7584 | 4.6994 | 51.33 | 84% |
| hist_gradient_numeric_25 | 6.4699 | 6.4685 | 8.8609 | 63.64 | 94% |
| router_without_freshness | 6.8197 | 6.8290 | 11.5917 | 111.81 | 90% |
| router_without_reset | 6.6769 | 6.6836 | 11.4569 | 114.00 | 92% |

The router fits experts on 25 devices and its gate on 5 more. The 30-device static baselines control for this additional label budget.

## Stress tests

| Condition | Static fusion MAE | Unconstrained router MAE | Constrained router MAE |
|---|---:|---:|---:|
| image_outage | 9.8604 | 10.3592 | 7.7799 |
| text_outage | 27.7643 | 21.1041 | 4.4719 |
| both_outage | 29.0856 | 26.4320 | 7.7022 |
| stale_history | 9.8604 | 11.3738 | 7.7799 |
| random_image_missing_50 | 8.4509 | 7.8886 | 7.5645 |
| block_image_outage | 7.9754 | 7.9080 | 7.0597 |
| curve_feature_noise | 7.2031 | 6.6938 | 6.7082 |

## Interpretation limits

- Same synthetic dataset previously explored; internal validation, not untouched external confirmation.
- Hyperparameters fixed before this run; no original-paper CNN reproduction.
- Bootstrap over devices is descriptive; models share training data across folds.
- Shifted stress tests do not inherit nominal conformal coverage guarantees.

Complete fold assignments, parameters, component definitions, widths, and descriptive paired-device bootstrap estimates are in the adjacent JSON files.
