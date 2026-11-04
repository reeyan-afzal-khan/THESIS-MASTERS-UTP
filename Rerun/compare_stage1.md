# Results comparison

- A (reference): `..\Results`
- B (re-run): `..\Rerun\Results`
- compared 50 files: close 2, different 43, identical 5
- only in A: 36 | only in B: 0

| file | verdict | detail |
|---|---|---|
| `cao/03_features/X_freq.npy` | different | 113534/1387620 values off (8.182%), max /d/ 0.347 |
| `cao/03_features/X_spatial.npy` | different | 23807/940498 values off (2.531%), max /d/ 6 |
| `cao/03_features/X_tf.npy` | close | max /d/ 9.54e-07 |
| `cao/03_features/X_time.npy` | close | max /d/ 1.86e-09 |
| `cao/03_features/extraction_log.csv` | different | chunk: 28/31 differ (e.g. 'chunk_00017' vs 'chunk_00018') (timing differs: seconds (max rel 2)) |
| `cao/03_features/extraction_timing.json` | different | per_domain_ms_per_window.freq: 3.339 vs 1.184; per_domain_ms_per_window.tf: 59.487 vs 19.695; per_domain_ms_per_window.spatial: 17.097 vs 6.034; per_domain_ms_per_window.TOTAL: 122.542 vs 36.231 (timing differs: wall_clock_seconds, windows_per_second, time) |
| `cao/03_features/sample_ids.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `cao/04_fusion/X_multi.npy` | different | 137341/15448836 values off (0.889%), max /d/ 6 |
| `cao/04_fusion/feature_names.json` | identical |  |
| `cao/04_fusion/feature_scales.csv` | different | std: 7/1002 rows off, max /d/ 0.000878; abs_max: 14/1002 rows off, max /d/ 0.0639 |
| `cao/04_fusion/manifest_aligned.parquet` | identical |  |
| `cao/04_fusion/sample_ids.npy` | identical |  |
| `cao/04_fusion/y.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `cao/05_selection/consensus_features.json` | different | keys differ: ['selected_features[35]', 'selected_features[36]', 'selected_features[37]', 'selected_features[38]'] |
| `cao/05_selection/domain_composition.csv` | different | n_selected: 4/4 rows off, max /d/ 20; pct_of_selected: 4/4 rows off, max /d/ 20.3; retention_pct: 4/4 rows off, max /d/ 9.84 |
| `cao/05_selection/fs_jaccard_multi.csv` | different | variance: 5/8 rows off, max /d/ 0.171; pearson: 3/8 rows off, max /d/ 0.0338; mutual_info: 4/8 rows off, max /d/ 0.0485; relieff: 5/8 rows off, max /d/ 0.0704 |
| `cao/05_selection/fs_masks_multi.npz` | different | pearson: 2/1002 entries differ; relieff: 2/1002 entries differ; mrmr: 200/1002 entries differ; rfe: 44/1002 entries differ |
| `cao/05_selection/fs_table_multi.csv` | different | feature: 981/1002 differ (e.g. 'TF:DWT_L4_entropy' vs 'SPAT:st_skew'); votes: 88/1002 rows off, max /d/ 1; score_variance: 933/1002 rows off, max /d/ 1.42e+13; rank_variance: 959/1002 rows off, max /d/ 937 |
| `cao/05_selection/fs_timings_multi.json` | different | variance: 0.03196510698762722 vs 0.017931600101292133; pearson: 0.3070550349948462 vs 0.20505480002611876; mutual_info: 50.86477717498201 vs 26.743097000056878; relieff: 216.2763281800144 vs 98.24354729987681 |
| `cao/05_selection/fs_votes_multi.npy` | different | 646/1002 values off (64.471%), max /d/ 3 |
| `cao/05_selection/mask_consensus.npy` | different | 64/1002 entries differ |
| `cao/05_selection/method_timings.csv` | different | method: 2/8 differ (e.g. 'shap' vs 'mutual_info') (timing differs: seconds (max rel 1.3)) |
| `cao/05_selection/rank_order.npy` | different | 986/1002 values off (98.403%), max /d/ 985 |
| `cao/05_selection/reduction_curve.csv` | different | accuracy: 12/12 rows off, max /d/ 0.118; balanced_accuracy: 12/12 rows off, max /d/ 0.146; f1_macro: 12/12 rows off, max /d/ 0.168 (timing differs: train_seconds (max rel 0.99)) |
| `cao/05_selection/selection_bias_analysis.csv` | different | cv_f1_macro: 3/3 rows off, max /d/ 2.08; ci95_low: 2/3 rows off, max /d/ 1.79; ci95_high: 2/3 rows off, max /d/ 2.36 |
| `tomasov/03_features/X_freq.npy` | different | 10941497/15521220 values off (70.494%), max /d/ 3.23e+07 |
| `tomasov/03_features/X_spatial.npy` | different | 5593253/10519938 values off (53.168%), max /d/ 2.42e+05 |
| `tomasov/03_features/X_tf.npy` | different | 119528595/127446462 values off (93.787%), max /d/ 9.32e+06 |
| `tomasov/03_features/X_time.npy` | different | 17044860/19315296 values off (88.245%), max /d/ 7.5e+03 |
| `tomasov/03_features/extraction_log.csv` | identical |  (timing differs: seconds (max rel 1.1)) |
| `tomasov/03_features/extraction_timing.json` | different | per_domain_ms_per_window.freq: 3.354 vs 1.124; per_domain_ms_per_window.tf: 60.672 vs 19.935; per_domain_ms_per_window.spatial: 37.403 vs 24.431; per_domain_ms_per_window.TOTAL: 144.594 vs 53.672 (timing differs: wall_clock_seconds, windows_per_second, time) |
| `tomasov/03_features/sample_ids.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `tomasov/04_fusion/X_multi.npy` | different | 153108205/172802916 values off (88.603%), max /d/ 3.23e+07 |
| `tomasov/04_fusion/feature_names.json` | identical |  |
| `tomasov/04_fusion/feature_scales.csv` | different | std: 5/1002 rows off, max /d/ 1; abs_max: 12/1002 rows off, max /d/ 0.0119 |
| `tomasov/04_fusion/manifest_aligned.parquet` | different | sample_id: 169366/172458 differ (e.g. 'tomasov::auto2_2023-04-17T124510+0100::b68::c215' vs 'tomasov::auto2_2023-04-17T124510+0100::b68::c141'); locus: 169286/172458 rows off, max /d/ 1.37e+03 |
| `tomasov/04_fusion/sample_ids.npy` | different | 169366/172458 entries differ |
| `tomasov/04_fusion/y.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `tomasov/05_selection/consensus_features.json` | different | keys differ: ['selected_features[52]', 'selected_features[53]'] |
| `tomasov/05_selection/domain_composition.csv` | different | n_selected: 3/4 rows off, max /d/ 3; pct_of_selected: 4/4 rows off, max /d/ 4.13; retention_pct: 3/4 rows off, max /d/ 2.22 |
| `tomasov/05_selection/fs_jaccard_multi.csv` | different | variance: 4/8 rows off, max /d/ 0.0564; pearson: 6/8 rows off, max /d/ 0.0226; mutual_info: 5/8 rows off, max /d/ 0.0252; relieff: 7/8 rows off, max /d/ 0.0525 |
| `tomasov/05_selection/fs_masks_multi.npz` | different | mutual_info: 14/1002 entries differ; relieff: 58/1002 entries differ; mrmr: 218/1002 entries differ; rfe: 186/1002 entries differ |
| `tomasov/05_selection/fs_table_multi.csv` | different | feature: 979/1002 differ (e.g. 'SPAT:st_vel_std' vs 'TF:tfband_2'); votes: 66/1002 rows off, max /d/ 1; score_variance: 925/1002 rows off, max /d/ 8.67e+13; rank_variance: 961/1002 rows off, max /d/ 950 |
| `tomasov/05_selection/fs_timings_multi.json` | different | variance: 0.38265566201880574 vs 0.2557343998923898; pearson: 4.938701770995976 vs 4.681295099901035; mutual_info: 89.24169001998962 vs 49.04942420008592; relieff: 230.30656925501535 vs 120.00509899994358 |
| `tomasov/05_selection/fs_votes_multi.npy` | different | 502/1002 values off (50.100%), max /d/ 3 |
| `tomasov/05_selection/mask_consensus.npy` | different | 32/1002 entries differ |
| `tomasov/05_selection/method_timings.csv` | different | method: 2/8 differ (e.g. 'rfe' vs 'mrmr') (timing differs: seconds (max rel 1.5)) |
| `tomasov/05_selection/rank_order.npy` | different | 988/1002 values off (98.603%), max /d/ 981 |
| `tomasov/05_selection/reduction_curve.csv` | different | accuracy: 7/12 rows off, max /d/ 0.0251; balanced_accuracy: 11/12 rows off, max /d/ 0.0172; f1_macro: 9/12 rows off, max /d/ 0.0171 (timing differs: train_seconds (max rel 1.8)) |
| `tomasov/05_selection/selection_bias_analysis.csv` | different | cv_f1_macro: 3/3 rows off, max /d/ 1.4; ci95_low: 2/3 rows off, max /d/ 1.16; ci95_high: 1/3 rows off, max /d/ 1.64 |

## Only in A
- `cao/04_fusion_smoke/X_multi.npy`
- `cao/04_fusion_smoke/feature_names.json`
- `cao/04_fusion_smoke/feature_scales.csv`
- `cao/04_fusion_smoke/manifest_aligned.parquet`
- `cao/04_fusion_smoke/sample_ids.npy`
- `cao/04_fusion_smoke/y.npy`
- `cao/05_selection_smoke/consensus_features.json`
- `cao/05_selection_smoke/domain_composition.csv`
- `cao/05_selection_smoke/fs_jaccard_multi.csv`
- `cao/05_selection_smoke/fs_masks_multi.npz`
- `cao/05_selection_smoke/fs_table_multi.csv`
- `cao/05_selection_smoke/fs_timings_multi.json`
- `cao/05_selection_smoke/fs_votes_multi.npy`
- `cao/05_selection_smoke/mask_consensus.npy`
- `cao/05_selection_smoke/method_timings.csv`
- `cao/05_selection_smoke/rank_order.npy`
- `cao/05_selection_smoke/reduction_curve.csv`
- `cao/05_selection_smoke/selection_bias_analysis.csv`
- `tomasov/04_fusion_smoke/X_multi.npy`
- `tomasov/04_fusion_smoke/feature_names.json`
- `tomasov/04_fusion_smoke/feature_scales.csv`
- `tomasov/04_fusion_smoke/manifest_aligned.parquet`
- `tomasov/04_fusion_smoke/sample_ids.npy`
- `tomasov/04_fusion_smoke/y.npy`
- `tomasov/05_selection_smoke/consensus_features.json`
- `tomasov/05_selection_smoke/domain_composition.csv`
- `tomasov/05_selection_smoke/fs_jaccard_multi.csv`
- `tomasov/05_selection_smoke/fs_masks_multi.npz`
- `tomasov/05_selection_smoke/fs_table_multi.csv`
- `tomasov/05_selection_smoke/fs_timings_multi.json`
- `tomasov/05_selection_smoke/fs_votes_multi.npy`
- `tomasov/05_selection_smoke/mask_consensus.npy`
- `tomasov/05_selection_smoke/method_timings.csv`
- `tomasov/05_selection_smoke/rank_order.npy`
- `tomasov/05_selection_smoke/reduction_curve.csv`
- `tomasov/05_selection_smoke/selection_bias_analysis.csv`
