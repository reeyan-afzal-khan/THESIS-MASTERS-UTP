# Results comparison

- A (reference): `..\Results`
- B (re-run): `..\Rerun\Results`
- compared 122 files: different 104, identical 18
- only in A: 62 | only in B: 0

| file | verdict | detail |
|---|---|---|
| `cao/00_inventory/class_summary.csv` | identical |  |
| `cao/00_inventory/manifest.parquet` | identical |  |
| `cao/00_inventory/official_split_leakage_audit.csv` | identical |  |
| `cao/01_splits/leakage_scorecard.csv` | identical |  |
| `cao/01_splits/split_manifest.parquet` | identical |  |
| `cao/01_splits/split_report.csv` | identical |  |
| `cao/04_fusion/X_multi.npy` | different | 14180125/15448836 values off (91.788%), max /d/ 3.09e+07 |
| `cao/04_fusion/feature_names.json` | identical |  |
| `cao/04_fusion/feature_scales.csv` | different | std: 7/1002 rows off, max /d/ 5; abs_max: 14/1002 rows off, max /d/ 0.0639 |
| `cao/04_fusion/manifest_aligned.parquet` | different | sample_id: 15400/15418 differ (e.g. 'cao::01_background::220112_cxm_background_01_single_data_32.mat' vs 'cao::01_background::220112_cxm_background_01_single_data_35.mat'); label: 9874/15418 differ (e.g. 'background' vs 'dig'); class_dir: 9874/15418 differ (e.g. '01_background' vs '02_dig'); official_split: 4948/15418 differ (e.g. 'test' vs 'train') |
| `cao/04_fusion/sample_ids.npy` | different | 15400/15418 entries differ |
| `cao/04_fusion/y.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `cao/05_selection/consensus_features.json` | different | keys differ: ['selected_features[53]', 'selected_features[54]', 'selected_features[55]', 'selected_features[56]'] |
| `cao/05_selection/domain_composition.csv` | different | n_selected: 4/4 rows off, max /d/ 16; pct_of_selected: 4/4 rows off, max /d/ 8.64; retention_pct: 4/4 rows off, max /d/ 11.5 |
| `cao/05_selection/fs_jaccard_multi.csv` | different | variance: 7/8 rows off, max /d/ 0.132; pearson: 7/8 rows off, max /d/ 0.175; mutual_info: 7/8 rows off, max /d/ 0.235; relieff: 7/8 rows off, max /d/ 0.0521 |
| `cao/05_selection/fs_masks_multi.npz` | different | variance: 18/1002 entries differ; pearson: 18/1002 entries differ; mutual_info: 18/1002 entries differ; relieff: 60/1002 entries differ |
| `cao/05_selection/fs_table_multi.csv` | different | shape (1002, 20) vs (1002, 22) |
| `cao/05_selection/fs_timings_multi.json` | different | variance: 0.03196510698762722 vs 0.021584799978882074; pearson: 0.3070550349948462 vs 0.21040410012938082; mutual_info: 50.86477717498201 vs 25.328533000079915; relieff: 216.2763281800144 vs 117.53709569992498 |
| `cao/05_selection/fs_votes_multi.npy` | different | 524/1002 values off (52.295%), max /d/ 7 |
| `cao/05_selection/mask_consensus.npy` | different | 43/1002 entries differ |
| `cao/05_selection/method_timings.csv` | different | method: 4/8 differ (e.g. 'relieff' vs 'rfe') (timing differs: seconds (max rel 1.1)) |
| `cao/05_selection/rank_order.npy` | different | 998/1002 values off (99.601%), max /d/ 990 |
| `cao/05_selection/reduction_curve.csv` | different | accuracy: 10/12 rows off, max /d/ 0.1; balanced_accuracy: 12/12 rows off, max /d/ 0.124; f1_macro: 11/12 rows off, max /d/ 0.134 (timing differs: train_seconds (max rel 1)) |
| `cao/05_selection/selection_bias_analysis.csv` | different | cv_f1_macro: 3/3 rows off, max /d/ 2.74; ci95_low: 2/3 rows off, max /d/ 3.18; ci95_high: 2/3 rows off, max /d/ 3.08 |
| `cao/06_benchmark/8_consensus/balancing_summary.csv` | different | strategy: 4/6 differ (e.g. 'smote_tomek' vs 'borderline_smote'); val_f1: 6/6 rows off, max /d/ 0.0482; val_balacc: 6/6 rows off, max /d/ 0.0387; test_f1: 6/6 rows off, max /d/ 0.0526 (timing differs: train_s (max rel 0.89)) |
| `cao/06_benchmark/8_consensus/benchmark_full.csv` | different | model: 27/42 differ (e.g. 'LightGBM' vs 'HistGradientBoosting'); strategy: 36/42 differ (e.g. 'smote' vs 'undersample'); val_accuracy: 42/42 rows off, max /d/ 0.0548; val_balanced_accuracy: 42/42 rows off, max /d/ 0.0563 (timing differs: train_seconds (max rel 7.1), test_ms_per_sample (max rel 94)) |
| `cao/06_benchmark/8_consensus/best_config.json` | different | n_features: 53 vs 76; val_f1_macro: 0.9355088657098586 vs 0.9706916807751488; test_accuracy: 0.9439822518025514 vs 0.9739323349972269; test_balanced_accuracy: 0.9336352203045492 vs 0.9728667145401256 (timing differs: train_seconds, inference_ms_per_sample) |
| `cao/06_benchmark/8_consensus/cv_best_config.csv` | different | accuracy: 5/5 rows off, max /d/ 0.0393; balanced_accuracy: 5/5 rows off, max /d/ 0.0355; precision_macro: 5/5 rows off, max /d/ 0.0412; recall_macro: 5/5 rows off, max /d/ 0.0355 (timing differs: train_seconds (max rel 1.1), infer_seconds (max rel 5.3), ms_per_sample (max rel 5.3)) |
| `cao/06_benchmark/8_consensus/domain_comparison.csv` | different | val_f1_macro: 5/5 rows off, max /d/ 0.402; test_accuracy: 5/5 rows off, max /d/ 0.61; test_bal_acc: 5/5 rows off, max /d/ 0.701; test_f1_macro: 5/5 rows off, max /d/ 0.731 (timing differs: train_seconds (max rel 1.8)) |
| `cao/06_benchmark/full_1002/balancing_summary.csv` | different | strategy: 6/6 differ (e.g. 'undersample' vs 'smote'); val_balacc: 1/6 rows off, max /d/ 0.00125; test_f1: 4/6 rows off, max /d/ 0.00413; val_f1: max /d/ 0.000765 (timing differs: train_s (max rel 1.3)) |
| `cao/06_benchmark/full_1002/benchmark_full.csv` | different | model: 13/42 differ (e.g. 'LightGBM' vs 'HistGradientBoosting'); strategy: 36/42 differ (e.g. 'undersample' vs 'class_weight'); val_accuracy: 20/42 rows off, max /d/ 0.011; val_balanced_accuracy: 25/42 rows off, max /d/ 0.0131 (timing differs: train_seconds (max rel 8.1), test_ms_per_sample (max rel 56)) |
| `cao/06_benchmark/full_1002/best_config.json` | different | best_strategy: 'undersample' vs 'class_weight'; test_accuracy: 0.9905712701053799 vs 0.9889073765945646; test_balanced_accuracy: 0.98979725807474 vs 0.9887254591996926; test_f1_macro: 0.9889653827503903 vs 0.9866339363580451 (timing differs: train_seconds, inference_ms_per_sample) |
| `cao/06_benchmark/full_1002/cv_best_config.csv` | different | accuracy: 3/5 rows off, max /d/ 0.00329; balanced_accuracy: 3/5 rows off, max /d/ 0.00348; precision_macro: 4/5 rows off, max /d/ 0.00365; recall_macro: 3/5 rows off, max /d/ 0.00348 (timing differs: train_seconds (max rel 1.6), infer_seconds (max rel 4), ms_per_sample (max rel 4)) |
| `cao/06_benchmark/full_1002/domain_comparison.csv` | different | best_strategy: 1/5 differ (e.g. 'none' vs 'class_weight'); val_f1_macro: 4/5 rows off, max /d/ 0.402; test_accuracy: 5/5 rows off, max /d/ 0.943; test_bal_acc: 5/5 rows off, max /d/ 1.31 (timing differs: train_seconds (max rel 1.6)) |
| `cao/07_final/8_consensus/class_ablation.csv` | different | dropped: 2/3 differ (e.g. 'walk' vs 'shake'); n_samples: 1/3 rows off, max /d/ 278; accuracy: 3/3 rows off, max /d/ 3; balanced_accuracy: 3/3 rows off, max /d/ 3.92 |
| `cao/07_final/8_consensus/confusion_counts.csv` | different | background: 5/6 rows off, max /d/ 22; dig: 3/6 rows off, max /d/ 6; knock: 3/6 rows off, max /d/ 4; shake: 6/6 rows off, max /d/ 8 |
| `cao/07_final/8_consensus/confusion_normalised.csv` | different | background: 5/6 rows off, max /d/ 0.0948; dig: 3/6 rows off, max /d/ 0.0162; knock: 3/6 rows off, max /d/ 0.0129; shake: 6/6 rows off, max /d/ 0.0522 |
| `cao/07_final/8_consensus/error_modes.csv` | different | shape (20, 4) vs (15, 4) |
| `cao/07_final/8_consensus/feature_importance.csv` | different | shape (53, 3) vs (76, 3) |
| `cao/07_final/8_consensus/headline_metrics.csv` | different | accuracy: 1/1 rows off, max /d/ 2.99; balanced_accuracy: 1/1 rows off, max /d/ 3.93; precision_macro: 1/1 rows off, max /d/ 3.72; recall_macro: 1/1 rows off, max /d/ 3.93 |
| `cao/07_final/8_consensus/model_card.json` | different | n_features: 53 vs 76; test.accuracy: 94.398 vs 97.393; test.balanced_accuracy: 93.364 vs 97.287; test.precision_macro: 93.228 vs 96.948 (timing differs: train_seconds, inference_ms_per_sample) |
| `cao/07_final/8_consensus/per_class_report.csv` | different | precision: 9/9 rows off, max /d/ 0.125; recall: 9/9 rows off, max /d/ 0.138; f1-score: 8/9 rows off, max /d/ 0.0953; support: 1/9 rows off, max /d/ 0.03 |
| `cao/07_final/full_1002/class_ablation.csv` | different | dropped: 1/3 differ (e.g. 'walk, water' vs 'walk, shake'); n_samples: 1/3 rows off, max /d/ 475; accuracy: 2/3 rows off, max /d/ 0.166; balanced_accuracy: 2/3 rows off, max /d/ 0.142 |
| `cao/07_final/full_1002/confusion_counts.csv` | different | dig: 1/6 rows off, max /d/ 1; knock: 1/6 rows off, max /d/ 3; shake: 2/6 rows off, max /d/ 2; walk: 2/6 rows off, max /d/ 1 |
| `cao/07_final/full_1002/confusion_normalised.csv` | different | dig: 1/6 rows off, max /d/ 0.0027; knock: 1/6 rows off, max /d/ 0.00482; shake: 2/6 rows off, max /d/ 0.00431; walk: 2/6 rows off, max /d/ 0.00431 |
| `cao/07_final/full_1002/error_modes.csv` | different | shape (10, 4) vs (12, 4) |
| `cao/07_final/full_1002/feature_importance.csv` | different | feature: 989/1002 differ (e.g. 'SPAT:st_t_entropy' vs 'SPAT:st_entropy'); importance: 613/1002 rows off, max /d/ 124; domain: 389/1002 differ (e.g. 'TF' vs 'SPAT') |
| `cao/07_final/full_1002/headline_metrics.csv` | different | accuracy: 1/1 rows off, max /d/ 0.17; balanced_accuracy: 1/1 rows off, max /d/ 0.11; precision_macro: 1/1 rows off, max /d/ 0.35; recall_macro: 1/1 rows off, max /d/ 0.11 |
| `cao/07_final/full_1002/model_card.json` | different | balancing: 'undersample' vs 'class_weight'; test.accuracy: 99.057 vs 98.891; test.balanced_accuracy: 98.98 vs 98.873; test.precision_macro: 98.818 vs 98.469 (timing differs: train_seconds, inference_ms_per_sample) |
| `cao/07_final/full_1002/per_class_report.csv` | different | precision: 6/9 rows off, max /d/ 0.0213; recall: 6/9 rows off, max /d/ 0.00482; f1-score: 8/9 rows off, max /d/ 0.0109; support: 1/9 rows off, max /d/ 0.00166 |
| `shared/08_crossdataset/agreement_statistics.json` | different | feature_selection_rank_spearman: 0.5509300224178536 vs 0.577461624882114; n_features_unanimous_on_both: 5 vs 14 |
| `shared/08_crossdataset/cao_leave_one_date_out.csv` | different | accuracy: 7/10 rows off, max /d/ 2.33; f1_macro: 8/10 rows off, max /d/ 1.15 |
| `shared/08_crossdataset/cross_system_transfer.csv` | different | accuracy: 3/4 rows off, max /d/ 3.25; balanced_accuracy: 3/4 rows off, max /d/ 3.4; f1_macro: 3/4 rows off, max /d/ 2.39 |
| `shared/08_crossdataset/features_stable_across_datasets.csv` | different | shape (5, 3) vs (14, 3) |
| `shared/08_crossdataset/parallel_domain_benchmark.csv` | different | test_accuracy: 8/10 rows off, max /d/ 0.61; test_f1_macro: 9/10 rows off, max /d/ 0.731 |
| `shared/08_crossdataset/tomasov_leave_one_recording_out.csv` | different | accuracy: 10/11 rows off, max /d/ 1.37; recall_on_true_class: 10/11 rows off, max /d/ 1.37 |
| `shared/09_report_8_consensus/tables/artefact_manifest.csv` | different | shape (666, 3) vs (426, 3) |
| `shared/09_report_8_consensus/tables/cnn_baseline.csv` | different | accuracy: 2/2 rows off, max /d/ 1.12; balanced_accuracy: 2/2 rows off, max /d/ 0.772; f1_macro: 2/2 rows off, max /d/ 1.86 (timing differs: train_seconds (max rel 0.84)) |
| `shared/09_report_8_consensus/tables/computational_efficiency.csv` | different | stage: 1/28 differ (e.g. 'LightGBM predict' vs 'XGBoost predict'); hours_for_100k_windows: 25/28 rows off, max /d/ 2.52 (timing differs: ms_per_window (max rel 9.8)) |
| `shared/09_report_8_consensus/tables/main_results.csv` | different | Features: 2/2 rows off, max /d/ 23; Model: 1/2 differ (e.g. 'LightGBM' vs 'XGBoost'); Accuracy (%): 1/2 rows off, max /d/ 3; Balanced acc. (%): 1/2 rows off, max /d/ 3.92 |
| `shared/09_report_8_consensus/tables/protocol_comparison.csv` | identical |  |
| `shared/09_report_8_consensus/tables/realtime_budget.csv` | different | real-time on 1 core: 1/2 differ (e.g. 'no' vs 'yes'); cores needed: 1/2 rows off, max /d/ 1 (timing differs: feature extraction (ms) (max rel 2.4), inference (ms) (max rel 9.8), total (ms) (max rel 2.4)) |
| `shared/09_report_8_consensus/tables/statistical_tests.csv` | different | friedman_chi2: 2/2 rows off, max /d/ 0.857; friedman_p: 1/2 rows off, max /d/ 3.65e-06; iman_davenport_F: 2/2 rows off, max /d/ 20.4; best_avg_rank: 1/2 rows off, max /d/ 0.334 |
| `shared/09_report_full_1002/tables/artefact_manifest.csv` | different | shape (646, 3) vs (434, 3) |
| `shared/09_report_full_1002/tables/cnn_baseline.csv` | different | accuracy: 2/2 rows off, max /d/ 1.9; balanced_accuracy: 1/2 rows off, max /d/ 2.07; f1_macro: 2/2 rows off, max /d/ 2.53 (timing differs: train_seconds (max rel 0.96)) |
| `shared/09_report_full_1002/tables/computational_efficiency.csv` | different | hours_for_100k_windows: 25/28 rows off, max /d/ 2.52 (timing differs: ms_per_window (max rel 5.6)) |
| `shared/09_report_full_1002/tables/main_results.csv` | different | Balancing: 2/2 differ (e.g. 'undersample' vs 'class_weight'); Accuracy (%): 2/2 rows off, max /d/ 0.166; Balanced acc. (%): 1/2 rows off, max /d/ 0.107; Macro-F1 (%): 1/2 rows off, max /d/ 0.234 |
| `shared/09_report_full_1002/tables/protocol_comparison.csv` | identical |  |
| `shared/09_report_full_1002/tables/realtime_budget.csv` | different | real-time on 1 core: 1/2 differ (e.g. 'no' vs 'yes'); cores needed: 1/2 rows off, max /d/ 1 (timing differs: feature extraction (ms) (max rel 2.4), inference (ms) (max rel 5.6), total (ms) (max rel 2.4)) |
| `shared/09_report_full_1002/tables/statistical_tests.csv` | different | friedman_chi2: 2/2 rows off, max /d/ 1.14; friedman_p: 1/2 rows off, max /d/ 2.37e-06; iman_davenport_F: 2/2 rows off, max /d/ 210; best_model: 1/2 differ (e.g. 'HistGradientBoosting' vs 'LightGBM') |
| `tomasov/00_inventory/acquisition_metadata.csv` | identical |  |
| `tomasov/00_inventory/class_summary.csv` | identical |  |
| `tomasov/00_inventory/manifest.parquet` | identical |  |
| `tomasov/00_inventory/recording_summary.csv` | identical |  |
| `tomasov/01_splits/R1C9_reconciliation.txt` | identical |  |
| `tomasov/01_splits/leakage_scorecard.csv` | identical |  |
| `tomasov/01_splits/split_manifest.parquet` | identical |  |
| `tomasov/01_splits/split_report.csv` | identical |  |
| `tomasov/04_fusion/X_multi.npy` | different | 154687177/172802916 values off (89.517%), max /d/ 3.28e+07 |
| `tomasov/04_fusion/feature_names.json` | identical |  |
| `tomasov/04_fusion/feature_scales.csv` | different | std: 21/1002 rows off, max /d/ 64.3; abs_max: 12/1002 rows off, max /d/ 0.0119 |
| `tomasov/04_fusion/manifest_aligned.parquet` | different | sample_id: 170014/172458 differ (e.g. 'tomasov::auto2_2023-04-17T124510+0100::b68::c215' vs 'tomasov::auto2_2023-04-17T124510+0100::b68::c141'); label: 93472/172458 differ (e.g. 'fence' vs 'construction'); stem: 93472/172458 differ (e.g. 'fence_2023-04-17T123117+0100' vs 'reconstruction1_2023-06-06T150714+0100'); block: 99918/172458 rows off, max /d/ 698 |
| `tomasov/04_fusion/sample_ids.npy` | different | 170014/172458 entries differ |
| `tomasov/04_fusion/y.npy` | different | could not compare: ValueError("Array can't be memory-mapped: Python objects in dtype.") |
| `tomasov/05_selection/consensus_features.json` | different | keys differ: ['selected_features[52]', 'selected_features[53]', 'selected_features[54]', 'selected_features[55]'] |
| `tomasov/05_selection/domain_composition.csv` | different | n_selected: 3/4 rows off, max /d/ 6; pct_of_selected: 4/4 rows off, max /d/ 2.43; retention_pct: 3/4 rows off, max /d/ 1.79 |
| `tomasov/05_selection/fs_jaccard_multi.csv` | different | variance: 6/8 rows off, max /d/ 0.0517; pearson: 6/8 rows off, max /d/ 0.17; mutual_info: 7/8 rows off, max /d/ 0.151; relieff: 7/8 rows off, max /d/ 0.0838 |
| `tomasov/05_selection/fs_masks_multi.npz` | different | variance: 18/1002 entries differ; pearson: 18/1002 entries differ; mutual_info: 26/1002 entries differ; relieff: 148/1002 entries differ |
| `tomasov/05_selection/fs_table_multi.csv` | different | shape (1002, 20) vs (1002, 22) |
| `tomasov/05_selection/fs_timings_multi.json` | different | variance: 0.38265566201880574 vs 0.29311790014617145; pearson: 4.938701770995976 vs 4.181357099907473; mutual_info: 89.24169001998962 vs 48.18266779999249; relieff: 230.30656925501535 vs 122.86446699989028 |
| `tomasov/05_selection/fs_votes_multi.npy` | different | 512/1002 values off (51.098%), max /d/ 7 |
| `tomasov/05_selection/mask_consensus.npy` | different | 33/1002 entries differ |
| `tomasov/05_selection/method_timings.csv` | different | method: 4/8 differ (e.g. 'boruta' vs 'rfe') (timing differs: seconds (max rel 1.5)) |
| `tomasov/05_selection/rank_order.npy` | different | 993/1002 values off (99.102%), max /d/ 946 |
| `tomasov/05_selection/reduction_curve.csv` | different | accuracy: 10/12 rows off, max /d/ 0.141; balanced_accuracy: 12/12 rows off, max /d/ 0.101; f1_macro: 11/12 rows off, max /d/ 0.101 (timing differs: train_seconds (max rel 3.1)) |
| `tomasov/05_selection/selection_bias_analysis.csv` | different | cv_f1_macro: 2/3 rows off, max /d/ 0.31; ci95_low: 2/3 rows off, max /d/ 0.74; ci95_high: 2/3 rows off, max /d/ 0.2 |
| `tomasov/06_benchmark/8_consensus/balancing_summary.csv` | different | val_f1: 4/6 rows off, max /d/ 0.006; val_balacc: 5/6 rows off, max /d/ 0.00554; test_f1: 5/6 rows off, max /d/ 0.00266 (timing differs: train_s (max rel 0.82)) |
| `tomasov/06_benchmark/8_consensus/benchmark_full.csv` | different | model: 24/42 differ (e.g. 'LightGBM' vs 'XGBoost'); strategy: 22/42 differ (e.g. 'smote' vs 'smote_tomek'); val_accuracy: 39/42 rows off, max /d/ 0.0691; val_balanced_accuracy: 38/42 rows off, max /d/ 0.0496 (timing differs: train_seconds (max rel 71), test_ms_per_sample (max rel 79)) |
| `tomasov/06_benchmark/8_consensus/best_config.json` | different | n_features: 52 vs 61; best_model: 'LightGBM' vs 'XGBoost'; test_f1_macro: 0.7358949507105252 vs 0.7370096570524284; cv_f1_ci95[0]: 0.6019504950998404 vs 0.6027832289247844 (timing differs: train_seconds, inference_ms_per_sample) |
| `tomasov/06_benchmark/8_consensus/cv_best_config.csv` | different | accuracy: 5/5 rows off, max /d/ 0.00353; balanced_accuracy: 5/5 rows off, max /d/ 0.0074; precision_macro: 5/5 rows off, max /d/ 0.00839; recall_macro: 5/5 rows off, max /d/ 0.0074 (timing differs: train_seconds (max rel 0.72), infer_seconds (max rel 12), ms_per_sample (max rel 12)) |
| `tomasov/06_benchmark/8_consensus/domain_comparison.csv` | different | val_f1_macro: 3/5 rows off, max /d/ 0.462; test_accuracy: 3/5 rows off, max /d/ 0.344; test_bal_acc: 3/5 rows off, max /d/ 0.161; test_f1_macro: 4/5 rows off, max /d/ 0.191 (timing differs: train_seconds (max rel 1.7)) |
| `tomasov/06_benchmark/full_1002/balancing_summary.csv` | different | strategy: 4/6 differ (e.g. 'borderline_smote' vs 'smote'); val_f1: 3/6 rows off, max /d/ 0.00243; val_balacc: 5/6 rows off, max /d/ 0.00291; test_f1: 6/6 rows off, max /d/ 0.0052 (timing differs: train_s (max rel 5.6)) |
| `tomasov/06_benchmark/full_1002/benchmark_full.csv` | different | model: 10/42 differ (e.g. 'LightGBM' vs 'HistGradientBoosting'); strategy: 20/42 differ (e.g. 'borderline_smote' vs 'class_weight'); val_accuracy: 34/42 rows off, max /d/ 0.0176; val_balanced_accuracy: 34/42 rows off, max /d/ 0.0199 (timing differs: train_seconds (max rel 21), test_ms_per_sample (max rel 18)) |
| `tomasov/06_benchmark/full_1002/best_config.json` | different | best_strategy: 'borderline_smote' vs 'class_weight'; val_f1_macro: 0.8668697157385923 vs 0.8654673667760863; test_accuracy: 0.9070937003705665 vs 0.9085053820363508; cv_f1_macro_mean: 0.7860290117324684 vs 0.7829025119187776 (timing differs: train_seconds, inference_ms_per_sample) |
| `tomasov/06_benchmark/full_1002/cv_best_config.csv` | different | accuracy: 3/5 rows off, max /d/ 0.00643; balanced_accuracy: 5/5 rows off, max /d/ 0.00711; precision_macro: 5/5 rows off, max /d/ 0.0098; recall_macro: 5/5 rows off, max /d/ 0.00711 (timing differs: train_seconds (max rel 5.9), infer_seconds (max rel 4.8), ms_per_sample (max rel 4.8)) |
| `tomasov/06_benchmark/full_1002/domain_comparison.csv` | different | val_f1_macro: 4/5 rows off, max /d/ 0.44; test_accuracy: 4/5 rows off, max /d/ 0.344; test_bal_acc: 3/5 rows off, max /d/ 0.188; test_f1_macro: 4/5 rows off, max /d/ 0.191 (timing differs: train_seconds (max rel 2)) |
| `tomasov/07_final/8_consensus/class_ablation.csv` | different | accuracy: 3/3 rows off, max /d/ 0.246; balanced_accuracy: 3/3 rows off, max /d/ 0.158; f1_macro: 3/3 rows off, max /d/ 0.255; f1_gain_pp: 2/3 rows off, max /d/ 0.3 |
| `tomasov/07_final/8_consensus/confusion_counts.csv` | different | car: 9/9 rows off, max /d/ 21; construction: 4/9 rows off, max /d/ 10; fence: 6/9 rows off, max /d/ 37; longboard: 9/9 rows off, max /d/ 9 |
| `tomasov/07_final/8_consensus/confusion_normalised.csv` | different | car: 9/9 rows off, max /d/ 0.0152; construction: 4/9 rows off, max /d/ 0.00448; fence: 6/9 rows off, max /d/ 0.0209; longboard: 9/9 rows off, max /d/ 0.0152 |
| `tomasov/07_final/8_consensus/error_modes.csv` | different | shape (42, 4) vs (39, 4) |
| `tomasov/07_final/8_consensus/feature_importance.csv` | different | shape (52, 3) vs (61, 3) |
| `tomasov/07_final/8_consensus/headline_metrics.csv` | different | precision_macro: 1/1 rows off, max /d/ 0.18; f1_macro: 1/1 rows off, max /d/ 0.11; accuracy: max /d/ 0.04; balanced_accuracy: max /d/ 0.06 |
| `tomasov/07_final/8_consensus/model_card.json` | different | model: 'LightGBM' vs 'XGBoost'; n_features: 52 vs 61; test.precision_macro: 76.323 vs 76.502; test.f1_macro: 73.589 vs 73.701 (timing differs: train_seconds, inference_ms_per_sample) |
| `tomasov/07_final/8_consensus/per_class_report.csv` | different | precision: 11/12 rows off, max /d/ 0.0211; recall: 8/12 rows off, max /d/ 0.0152; f1-score: 9/12 rows off, max /d/ 0.0112; support: max /d/ 0.000353 |
| `tomasov/07_final/full_1002/class_ablation.csv` | different | accuracy: 2/3 rows off, max /d/ 0.15; balanced_accuracy: 1/3 rows off, max /d/ 0.161; f1_macro: 1/3 rows off, max /d/ 0.134; f1_gain_pp: 2/3 rows off, max /d/ 0.17 |
| `tomasov/07_final/full_1002/confusion_counts.csv` | different | car: 5/9 rows off, max /d/ 13; construction: 4/9 rows off, max /d/ 6; fence: 5/9 rows off, max /d/ 32; longboard: 3/9 rows off, max /d/ 15 |
| `tomasov/07_final/full_1002/confusion_normalised.csv` | different | car: 5/9 rows off, max /d/ 0.0455; construction: 4/9 rows off, max /d/ 0.00339; fence: 5/9 rows off, max /d/ 0.0288; longboard: 3/9 rows off, max /d/ 0.0189 |
| `tomasov/07_final/full_1002/error_modes.csv` | different | true: 25/34 differ (e.g. 'regular' vs 'openclose'); predicted: 26/34 differ (e.g. 'fence' vs 'walk'); pct_of_true: 29/34 rows off, max /d/ 4.55; n: 25/34 rows off, max /d/ 165 |
| `tomasov/07_final/full_1002/feature_importance.csv` | different | feature: 981/1002 differ (e.g. 'SPAT:st_entropy' vs 'SPAT:st_edge_density_t'); importance: 837/1002 rows off, max /d/ 1.95e+03; domain: 339/1002 differ (e.g. 'FREQ' vs 'SPAT') |
| `tomasov/07_final/full_1002/headline_metrics.csv` | different | accuracy: 1/1 rows off, max /d/ 0.14; f1_weighted: 1/1 rows off, max /d/ 0.16; auc_ovr_macro: 1/1 rows off, max /d/ 0.29; balanced_accuracy: max /d/ 0.04 |
| `tomasov/07_final/full_1002/model_card.json` | different | balancing: 'borderline_smote' vs 'class_weight'; test.accuracy: 90.709 vs 90.851; test.f1_weighted: 90.074 vs 90.229; top_error_mode.pct_of_true: 77.77777777777779 vs 73.23232323232324 (timing differs: train_seconds, inference_ms_per_sample) |
| `tomasov/07_final/full_1002/per_class_report.csv` | different | precision: 11/12 rows off, max /d/ 0.0123; recall: 9/12 rows off, max /d/ 0.0306; f1-score: 11/12 rows off, max /d/ 0.0155; support: 1/12 rows off, max /d/ 0.00141 |

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
- `cao/06_benchmark_smoke/balancing_summary.csv`
- `cao/06_benchmark_smoke/benchmark_full.csv`
- `cao/06_benchmark_smoke/best_config.json`
- `cao/06_benchmark_smoke/cv_best_config.csv`
- `cao/06_benchmark_smoke/domain_comparison.csv`
- `cao/07_final_smoke/class_ablation.csv`
- `cao/07_final_smoke/confusion_counts.csv`
- `cao/07_final_smoke/confusion_normalised.csv`
- `cao/07_final_smoke/error_modes.csv`
- `cao/07_final_smoke/feature_importance.csv`
- `cao/07_final_smoke/headline_metrics.csv`
- `cao/07_final_smoke/model_card.json`
- `cao/07_final_smoke/per_class_report.csv`
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
- `tomasov/06_benchmark_smoke/balancing_summary.csv`
- `tomasov/06_benchmark_smoke/benchmark_full.csv`
- `tomasov/06_benchmark_smoke/best_config.json`
- `tomasov/06_benchmark_smoke/cv_best_config.csv`
- `tomasov/06_benchmark_smoke/domain_comparison.csv`
- `tomasov/07_final_smoke/class_ablation.csv`
- `tomasov/07_final_smoke/confusion_counts.csv`
- `tomasov/07_final_smoke/confusion_normalised.csv`
- `tomasov/07_final_smoke/error_modes.csv`
- `tomasov/07_final_smoke/feature_importance.csv`
- `tomasov/07_final_smoke/headline_metrics.csv`
- `tomasov/07_final_smoke/model_card.json`
- `tomasov/07_final_smoke/per_class_report.csv`
