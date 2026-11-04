# Stage 4 - notebooks 14-15: Jetson `Results/` vs re-run `Rerun/Results/`

Re-run on the Windows laptop (RTX 4060 Laptop, Python 3.13.9) with the reproducible
feature selection, 2026-10-30: notebook 14 took 88.5 min, notebook 15 31.2 min.

## Reading the raw diff

The file-level diff at the bottom marks all 15 files "different".  Matched by key,
the differences have four causes, none of them a reproducibility failure:

1. **Row order.** `fusion.fuse()` now orders rows by the manifest; the Jetson's
   `X_multi` was in extraction order.  `headline_predictions_*.csv` stores rows
   without an id, so its rows no longer line up.  Matched by `sample_id`, `y_true`
   agrees on 100% of test windows.
2. **Position-based subsamples.** Notebook 14 draws 8,000 (repeated splits) and
   3,000 (leave-one-out) windows per class by row position; notebook 15 draws its
   arm-B random split and its 12,000-per-class selection subsample the same way.
   A new row order gives a different, equally valid draw.  Cao's repeated splits
   use every row and are unaffected (`n_train` identical).
3. **New selection rules.** Notebook 15 refits the selection ensemble with
   deterministic tie-breaking and without the 36 unstable features.
4. **Hard-coded headline.** Notebook 14 fixes `HEADLINE` to the Jetson's winners
   (Tomasov LightGBM/borderline_smote, Cao LightGBM/undersample) and
   `within = 99.06`.  The re-run's notebook 07 chose LightGBM/class_weight on both
   (Tomasov 90.85/82.01, Cao 98.89/98.66), so this re-run bootstraps the Jetson's
   configurations, not its own headline.

## Notebook 14 - statistics

### Repeated partitions (macro-F1 over 5 seeds, mean +/- SD (range))

| dataset | condition | Jetson | re-run |
|---|---|---|---|
| Tomasov | LightGBM / class_weight | 80.47 +/- 4.42 (74.52-84.99) | 80.17 +/- 3.13 (75.17-83.69) |
| Tomasov | RandomForest / class_weight | 72.79 +/- 4.25 | 72.47 +/- 2.54 |
| Tomasov | LogisticRegression / class_weight | 73.14 +/- 5.22 | 74.78 +/- 2.63 |
| Cao | LightGBM / class_weight | 99.13 +/- 0.30 | 99.09 +/- 0.35 |
| Cao | RandomForest / class_weight | 96.95 +/- 1.79 | 96.92 +/- 1.76 |
| Cao | LogisticRegression / class_weight | 97.59 +/- 0.78 | 97.65 +/- 0.88 |

LightGBM leads both other learners on 5/5 seeds, on both datasets, in both runs.
`statistics_verdict.json` reports the largest SD: Tomasov 5.22 -> 3.13 (logistic
regression on the Jetson, LightGBM in the re-run), Cao 1.79 -> 1.76.

### Group bootstrap, same configuration (95% half-width)

| | Jetson | re-run |
|---|---|---|
| Tomasov accuracy: window / zone / recording | +/-0.52 / 5.89 / 7.58 | +/-0.54 / 5.78 / 7.48 |
| Cao accuracy: window / session / date | +/-0.42 / 0.58 / 0.52 | +/-0.50 / 0.70 / 0.54 |
| Tomasov point acc / F1 | 90.71 / 81.98 | 90.49 / 81.96 |
| Cao point acc / F1 | 99.06 / 98.90 | 98.89 / 98.73 |

The same three macro-F1 levels are not estimable in both runs.  Matched by
`sample_id`, the headline predictions agree on 97.94% (Tomasov) and 99.61% (Cao)
of test windows.

### Paired Wilcoxon over held-out acquisitions, Holm-corrected

| | Jetson | re-run |
|---|---|---|
| significant pairs | 2 of 45 | 4 of 45 |
| Tomasov | undersample > class_weight (+5.13, p 0.041); borderline_smote > class_weight (+1.44, p 0.029) | undersample > none (+7.33, p 0.047), > class_weight (+6.37, p 0.029), > smote_tomek (+5.91, p 0.027), > smote (+5.77, p 0.043) |
| Cao | none (smallest corrected p 0.82; strategies span 0.45 pts) | none (smallest corrected p 0.22; span 0.69 pts) |

Only "undersampling beats class weighting on Tomasov" is significant in both runs.
All Tomasov corrected p-values sit between 0.027 and 0.047 with 11 held-out
recordings, so the count of significant pairs depends on the subsample.

### Cao leave-one-date-out reconciliation

Mean accuracy 86.81 -> 86.48; union macro-F1 57.82 -> 57.75; drop against the
hard-coded 99.06: 12.25 -> 12.58 (12.41 against the re-run's own 98.89).

## Notebook 15 - protocol attribution (LightGBM / class_weight)

### Reproduced: full representation, dimensionality reduction, consensus

| config | Tomasov A -> B, Jetson | Tomasov, re-run | Cao, Jetson | Cao, re-run |
|---|---|---|---|---|
| full_1002 | 90.82 -> 97.65 (+6.84) | 90.85 -> 97.85 (+7.00) | 99.06 -> 99.29 (+0.23) | 98.89 -> 99.48 (+0.59) |
| LDA | 82.24 -> 83.90 (+1.66) | 82.28 -> 84.48 (+2.20) | 99.00 -> 99.09 (+0.09) | 99.00 -> 98.90 (-0.10) |
| PCA_95 | 71.93 -> 73.24 (+1.31) | 72.16 -> 73.22 (+1.06) | 96.84 -> 96.89 (+0.05) | 96.84 -> 97.60 (+0.76) |
| 8_consensus | 77.78 -> 80.20 (+2.42) | 77.71 -> 80.34 (+2.63) | 96.06 -> 95.92 (-0.15) | 97.34 -> 98.06 (+0.72) |

### Changed: three Tomasov top-100 subsets (accuracy, A blocked -> B random)

| selector | Jetson | re-run | cause |
|---|---|---|---|
| Boruta | 64.10 -> 59.81 (-4.29) | 80.84 -> 87.63 (+6.80) | the Jetson cut 100 features from 763 tied at rank 1 |
| RFE | 78.40 -> 77.66 (-0.74) | 80.51 -> 87.11 (+6.60) | the Jetson cut inside a tied elimination block |
| SHAP | 90.90 -> 97.63 (+6.72) | 80.99 -> 87.09 (+6.10) | 8 spatial-profile features now excluded (check below) |

The other five selectors move by at most 1.2 points in arm A.  Median Tomasov
inflation is 4.09 -> 6.19 (range -4.29..6.84 -> 0.52..7.00); Cao 0.35 -> 0.79
(largest 2.13 -> 1.66).  Every Tomasov configuration now gains under the random
split.  Across arms, Tomasov Boruta's top-100 Jaccard rises 0.28 -> 0.80 (the
Jetson's low value came from the tie cut); RFE stays the least stable (0.23 ->
0.28).

### SHAP check: does the exclusion cause the Tomasov drop?

Notebook 15, arm A, replicated exactly (same 12,000-per-class subsample, SHAP
scorer and seed), with LightGBM/class_weight refitted on the full blocked split:

| feature set | n | accuracy | macro-F1 |
|---|---|---|---|
| SHAP top-100, unstable excluded (as notebook 15) | 100 | 80.93 | 76.37 |
| SHAP top-100, all 1,002 eligible | 100 | 90.44 | 81.36 |
| unrestricted SHAP top-100 minus its 8 unstable | 92 | 81.25 | 76.95 |

The unrestricted top-100 contains 8 of the 36 features, all spatial-profile
statistics: `SPAT:sp_gini` (rank 12), `sp_grad_energy` (20), `sp_tv_norm` (21),
`sp_grad_mean_abs` (22), `sp_grad_std` (23), `sp_grad_max` (24), `sp_skew` (33)
and `sp_kurt` (56).  Removing just those eight takes the subset from 90.44 to
81.25: the exclusion explains the drop.  None of the 36 reached 8 votes on the
Jetson (at most 7 on either dataset), so the consensus never contained them and
the exclusion does not touch the consensus results of notebooks 06-07.

## Finding: the full models lean on numerically degenerate features

The same refits on the full blocked/grouped splits (LightGBM / class_weight,
re-run data; SP8 = the eight spatial-profile features above):

| dataset | full 1,002 | minus SP8 (994) | minus all 36 (966) | SP8 alone (8) |
|---|---|---|---|---|
| Tomasov acc / F1 | 90.85 / 82.01 | 83.82 / 79.94 | 81.24 / 76.82 | 61.13 / 48.53 |
| Cao acc / F1 | 98.89 / 98.66 | 98.56 / 98.28 | 98.45 / 98.20 | 71.49 / 68.77 |

Mechanism.  `preprocess_patch` z-scores every channel as
`(x - mean) / (sd + 1e-12)` before extraction, and `features_spatial.extract`
builds the channel-energy profile `E = (X**2).sum(axis=1)` from that patch.  In
exact arithmetic every channel then has `E_c = T`, so the profile is flat by
construction; the 1e-12 term leaves `E_c / T - 1 = -2e-12 / sd_c`, an inverted
imprint of each channel's band-passed amplitude at about 1e-14.  Measured on one
window from each of six Tomasov recordings, `E/T - 1` correlates with
`-2e-12/sd_c` at +0.9993 to +0.9998 (raw sd 48-1,980, deviations 7e-16 to
4e-14, against rounding near 1e-15).  Trees split on order, not scale, so the
imprint is usable; its closeness to rounding is why these features fail the
perturbation test on 85-97% of windows.  The full-1,002 models still reproduce
across machines (Tomasov 90.71 vs 90.85) because the imprint, not the rounding,
dominates - but it would vanish under float32 or a different epsilon.  The other
28 excluded features (`TIME:mean`, `SPAT:st_mean`, spectrum minima, empty
histogram bins, some cepstral terms) are degenerate for the same reason - a
statistic of a normalised signal that is constant by construction - and add a
further 2.6 accuracy points on Tomasov.

## File-level diff (unaligned, from `_compare_results.py`)

| file | verdict | detail |
|---|---|---|
| `shared/14_statistics/cao_lodo_reconciliation.json` | different | published_macro_f1_union: 57.82 vs 57.75; mean_accuracy: 86.81 vs 86.48; accuracy_drop_pp: 12.25 vs 12.58 |
| `shared/14_statistics/group_bootstrap_ci.csv` | different | point: 9/12 rows off, max /d/ 0.22; ci_low: 10/12 rows off, max /d/ 0.524; ci_high: 7/12 rows off, max /d/ 0.363; half_width: 12/12 rows off, max /d/ 0.281 |
| `shared/14_statistics/headline_predictions_cao.csv` | different | y_true: 984/1803 differ (e.g. 'background' vs 'dig'); y_pred: 990/1803 differ (e.g. 'background' vs 'dig') |
| `shared/14_statistics/headline_predictions_tomasov.csv` | different | y_true: 4943/11334 differ (e.g. 'fence' vs 'construction'); y_pred: 5777/11334 differ (e.g. 'car' vs 'regular') |
| `shared/14_statistics/leave_one_acquisition_out_scores.csv` | different | accuracy: 148/186 rows off, max /d/ 20.4; balanced_accuracy: 145/186 rows off, max /d/ 20.4; f1_macro: 39/186 rows off, max /d/ 3.47; f1_macro_union: 142/186 rows off, max /d/ 50.1 |
| `shared/14_statistics/paired_wilcoxon_holm.csv` | different | condition_a: 35/45 differ (e.g. 'LightGBM / borderline_smote' vs 'LightGBM / smote_tomek'); condition_b: 25/45 differ (e.g. 'LightGBM / class_weight' vs 'LightGBM / undersample'); mean_diff: 45/45 rows off, max /d/ 9.46; median_diff: 39/45 rows off, max /d/ 5.61 |
| `shared/14_statistics/repeated_splits_raw.csv` | different | n_train: 15/30 rows off, max /d/ 897; n_test: 15/30 rows off, max /d/ 342; accuracy: 21/30 rows off, max /d/ 9.16; balanced_accuracy: 22/30 rows off, max /d/ 8.25 |
| `shared/14_statistics/repeated_splits_summary.csv` | different | mean: 3/6 rows off, max /d/ 1.63; sd: 6/6 rows off, max /d/ 2.6; min: 5/6 rows off, max /d/ 7.11; max: 4/6 rows off, max /d/ 1.69 |
| `shared/14_statistics/statistics_verdict.json` | different | n_significant_holm: 2 vs 4; repeat_sd_f1_macro.tomasov: 5.221 vs 3.126; repeat_sd_f1_macro.cao: 1.792 vs 1.756; bootstrap_widening.tomasov.window (naive): 1.24 vs 1.266 |
| `shared/15_protocol_attribution/cnn_baseline_sources.json` | different | [0].table[0].released 8:2 (%): 91.31 vs 92.02; [0].table[0].session-grouped (%): 91.29 vs 92.18; [0].table[0].drop (pp): 0.01 vs -0.16; [0].table[1].released 8:2 (%): 90.85 vs 91.63 |
| `shared/15_protocol_attribution/config_sizes.csv` | different | n_features: 4/40 rows off, max /d/ 26 |
| `shared/15_protocol_attribution/dimreduction_arms_raw.csv` | different | n_features: 1/8 rows off, max /d/ 1; accuracy: 4/8 rows off, max /d/ 0.713; balanced_accuracy: 4/8 rows off, max /d/ 0.738; f1_macro: 4/8 rows off, max /d/ 0.715 |
| `shared/15_protocol_attribution/feature_arms_raw.csv` | different | n_features: 4/40 rows off, max /d/ 26; accuracy: 35/40 rows off, max /d/ 27.8; balanced_accuracy: 37/40 rows off, max /d/ 26.3; f1_macro: 37/40 rows off, max /d/ 27.2 |
| `shared/15_protocol_attribution/protocol_inflation.csv` | different | config: 22/24 differ (e.g. 'top100_boruta' vs 'top100_variance'); acc_A_blocked: 23/24 rows off, max /d/ 10.3; acc_B_random: 22/24 rows off, max /d/ 13.4; f1_A_blocked: 23/24 rows off, max /d/ 8.96 |
| `shared/15_protocol_attribution/selector_stability_across_arms.csv` | different | overlap: 14/18 rows off, max /d/ 45; jaccard: 14/18 rows off, max /d/ 0.52 |
