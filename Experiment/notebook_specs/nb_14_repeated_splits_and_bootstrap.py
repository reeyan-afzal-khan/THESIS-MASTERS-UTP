"""Notebook 14 - replace the Friedman/Nemenyi analysis (reviewer R2C5).

The submitted paper ran a Friedman test over six balancing strategies as blocks
and seven classifiers as treatments, on a single partition of a single dataset.
The six strategies are not independent experimental units: they share every
sample, the same partition, and the same feature matrix.  This notebook replaces
that analysis with three tests whose units really are independent, in increasing
order of strength.
"""

CELLS = [
    ("md", """
# 14 - Repeated Splits, Group Bootstrap, and Paired Tests over Acquisitions

## The objection (R2C5)

Table 14 reports a Friedman test with chi-square 35.43 and p = 3.6e-06, computed over
**six balancing strategies as blocks** and seven classifiers as treatments, on one
partition of one dataset. The reviewer is right that this is not a valid application of
the Demsar framework:

* The six strategies are not independent replicates. They resample **the same training
  rows**, evaluated on **the same test rows**, from **the same partition**. Almost every
  source of variance is shared, so the residual differences are far smaller than the test
  assumes, and the p-value is correspondingly overstated.
* A single partition gives no estimate of partition variance at all. If a different seed
  would have moved macro-F1 by three points, the reported ranking is noise.
* Window-level confidence intervals treat 11,334 windows that overlap by 75% as 11,334
  independent observations, which understates their width by a large factor.

## What replaces it

| # | Test | Unit of replication | Answers |
|---|---|---|---|
| 1 | Repeated group-aware splits | one partition, five seeds | how much does the result move if the cut falls elsewhere? |
| 2 | Group-level bootstrap | one recording / acquisition date | how wide is the interval once dependence is respected? |
| 3 | Paired Wilcoxon over held-out acquisitions | one held-out recording / date | is one condition really better, on units that are independent by construction? |

Test 3 is the direct replacement for the Friedman test: same six balancing strategies,
same classifier, but blocked on **separate acquisitions** instead of on configurations of
one acquisition.

## A documented approximation

Features exist only for the rows that survived the guard band and decimation of the
original partition. Re-splitting with a different seed moves the guard band, so a strictly
correct repeat would need re-extraction. We instead re-partition **the rows that already
have features**, which keeps the population fixed across repeats and isolates partition
variance. This is stated in the paper; it slightly understates variability, because the
population itself was chosen under seed 42.
"""),
    ("code", """
import sys, json, time
from pathlib import Path
# No machine-specific path: the kernel starts in Experiment/ (or the repo root).
EXPERIMENT_DIR = Path.cwd() if (Path.cwd() / "dasfe").is_dir() else Path.cwd() / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dasfe import config as C, fusion as F, evaluate as EV, robust as R

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
DATASETS   = ("tomasov", "cao")
FEATURE_SET = "full_1002"          # matches the headline row of main_results.csv

# The configuration each dataset reports as its headline result is read
# from notebook 07's model card (below), so the bootstrap follows whatever
# that run selected instead of a value copied from an earlier run.

# Test 1: which conditions to track across repeated partitions.
REPEAT_CONFIGS = [("LightGBM", "class_weight"),
                  ("RandomForest", "class_weight"),
                  ("LogisticRegression", "class_weight")]
N_REPEATS  = 5

# Test 3: the six balancing strategies, one classifier - the direct replacement
# for the retired Friedman test.
PAIRED_CONFIGS = [("LightGBM", s) for s in C.BALANCING_STRATEGIES]

# Grouping columns.  The first is the leave-one-out unit; the bootstrap ladder
# runs from the naive window level up to it.
GROUP_COLS = {"tomasov": {"loo": "recording_id",
                          "ladder": ("zone_id", "recording_id")},
              "cao":     {"loo": "date_id",
                          "ladder": ("session_id", "date_id")}}

N_BOOT     = 2000
# Stratified caps.  None uses every row; the repeated-split and leave-one-out
# sweeps refit many times, so they are capped to stay tractable on the Jetson.
# At 8000/class the Tomasov repeated-split sweep took 3,870 s for 15 fits, so
# the leave-one-out sweep (6 strategies x 13 recordings = 78 fits) is capped
# harder: 6000 would cost roughly 4.3 hours, 3000 roughly 2.
REPEAT_PER_CLASS = 8000
LOO_PER_CLASS    = 3000

# Reuse the expensive sweeps when their CSV already exists.  Set False to force
# a recomputation.  This exists because the first run crashed in the plotting
# cell after the 65-minute repeated-split sweep had already succeeded.
REUSE_CACHED = True
# ------------------------------------------------------------------------

out_dir = C.results_dir("shared", "14_statistics")
print("output:", out_dir)

CARDS = {ds: json.loads((C.results_dir(ds, "07_final", FEATURE_SET) / "model_card.json")
                        .read_text(encoding="utf-8")) for ds in DATASETS}
HEADLINE = {ds: (c["model"], c["balancing"]) for ds, c in CARDS.items()}
print("headline:", HEADLINE)

DATA = {}
for ds in DATASETS:
    d = F.load(C.results_dir(ds, "04_fusion"))
    d["manifest"] = d["manifest"].reset_index(drop=True)
    d["split"] = d["manifest"]["split"].to_numpy()
    DATA[ds] = d
    print(f"{ds:9s} X {d['X'].shape}  {len(set(d['y']))} classes  "
          f"{d['manifest'][GROUP_COLS[ds]['loo']].nunique()} "
          f"{GROUP_COLS[ds]['loo']} units")
"""),
    ("md", """
## 0. The test being retired

Restated here so the revision is auditable: what was claimed, and why it does not hold.
"""),
    ("code", """
old_path = (C.RESULTS_DIR / "shared" / f"09_report_{FEATURE_SET}" / "tables"
            / "statistical_tests.csv")
if old_path.exists():
    old = pd.read_csv(old_path)
    display(old)
    print("Blocks: the six balancing strategies. Treatments: the seven classifiers.")
    print("Every block reuses the same rows, the same partition and the same features,")
    print("so the blocks are not independent replicates and the p-value is not valid.")
else:
    old = None
    print("previous statistical_tests.csv not found at", old_path)
"""),
    ("md", """
## 1. Repeated group-aware partitions

Five partitions of the same population, each built with the same leakage-safe rule and a
different seed. Tomasov re-cuts equal-mass contiguous segments per recording and re-applies
the guard band; Cao re-draws session-grouped folds. The spread across seeds is the
partition variance the single-split table could not show.
"""),
    ("code", """
raw_path = out_dir / "repeated_splits_raw.csv"
if REUSE_CACHED and raw_path.exists():
    rep_all = pd.read_csv(raw_path)
    print(f"reusing {raw_path.name} ({len(rep_all)} rows). "
          f"Set REUSE_CACHED = False to recompute.")
else:
    parts = []
    for ds in DATASETS:
        d = DATA[ds]
        idx = R.subsample_index(d["y"], REPEAT_PER_CLASS, C.SEED)
        X, y = d["X"][idx], d["y"][idx]
        mani = d["manifest"].iloc[idx].reset_index(drop=True)
        print(f"=== {ds}: {len(idx):,} rows, {N_REPEATS} partitions ===")
        t0 = time.perf_counter()
        rep = R.run_repeated(X, y, mani, ds, REPEAT_CONFIGS,
                             n_repeats=N_REPEATS, base_seed=C.SEED)
        rep["dataset"] = ds
        parts.append(rep)
        print(f"  {time.perf_counter()-t0:.0f}s")
    rep_all = pd.concat(parts, ignore_index=True)
    rep_all.to_csv(raw_path, index=False)

summ = []
for ds in DATASETS:
    s = R.repeat_summary(rep_all[rep_all.dataset == ds]).reset_index()
    s.insert(0, "dataset", ds)
    summ.append(s)
summ = pd.concat(summ, ignore_index=True)
display(summ)
summ.to_csv(out_dir / "repeated_splits_summary.csv", index=False)
"""),
    ("code", """
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.2 * len(DATASETS), 4.2),
                         squeeze=False)
for ax, ds in zip(axes[0], DATASETS):
    sub = rep_all[rep_all.dataset == ds]
    sns.boxplot(data=sub, x="condition", y="f1_macro", ax=ax, width=0.5,
                showfliers=False, color="#bdd7e7")
    sns.stripplot(data=sub, x="condition", y="f1_macro", ax=ax,
                  color="#08519c", size=5, jitter=0.12)
    ax.set_title(f"{ds}: macro-F1 over {N_REPEATS} partitions")
    ax.set_xlabel("")
    ax.set_ylabel("macro-F1 (%)")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
fig.savefig(out_dir / "repeated_splits.png", dpi=200, bbox_inches="tight")
plt.show()
"""),
    ("md", """
## 2. Group-level bootstrap on the reported test set

The headline configuration is refitted on the stored partition and its test predictions
are resampled at three levels. The window level is the naive interval the original
submission implied; the coarser levels are the honest ones. The width should grow
substantially as the resampling unit becomes a real acquisition unit.

**Which classes enter the macro average.** A replicate that misses a recording also misses
every class unique to it. Scoring those classes as zero would not measure performance, it
would measure the resampling scheme: on Tomasov five of nine classes come from a single
recording each, and with thirteen test recordings a given one is missed about 35% of the
time, which is enough to push the point estimate outside its own interval. Macro-F1 is
therefore averaged over the classes each replicate contains, and
`frac_replicates_all_classes` records how often that is the full set.

**When the interval is not usable.** The `estimable` column is False when the point
estimate falls outside the interval or when fewer than 90% of replicates contain every
class. On this data that is expected for Tomasov macro-F1 at recording level: thirteen
acquisitions cannot support a macro average over nine classes. Report accuracy at
recording level and macro-F1 at zone level, and say why - the same sparsity of independent
acquisitions is what the confound analysis is about.
"""),
    ("code", """
boot_rows, preds = [], {}
for ds in DATASETS:
    d = DATA[ds]
    model, strategy = HEADLINE[ds]
    sp = d["split"]
    tr, te = sp == "train", sp == "test"
    print(f"=== {ds}: {model} / {strategy}  ({tr.sum():,} train, {te.sum():,} test) ===")
    pred_path = out_dir / f"headline_predictions_{ds}.csv"
    if REUSE_CACHED and pred_path.exists():
        cached = pd.read_csv(pred_path)
        y_true, y_pred = cached["y_true"].to_numpy(), cached["y_pred"].to_numpy()
        print(f"  reusing {pred_path.name}")
    else:
        r = EV.fit_predict(model, strategy, d["X"][tr], d["y"][tr],
                           d["X"][te], d["y"][te], C.SEED)
        y_true, y_pred = d["y"][te], r["y_pred"]
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(pred_path, index=False)
        print(f"  accuracy {100*r['metrics']['accuracy']:.2f}%  "
              f"macro-F1 {100*r['metrics']['f1_macro']:.2f}%")
    preds[ds] = (y_true, y_pred, d["manifest"][te])

    mte = d["manifest"][te]
    levels = {"window (naive)": np.arange(int(te.sum()))}
    for col in GROUP_COLS[ds]["ladder"]:
        levels[f"{col}"] = mte[col].to_numpy()
    for metric in ("f1_macro", "accuracy"):
        t = R.bootstrap_by_level(y_true, y_pred, levels,
                                 n_boot=N_BOOT, seed=C.SEED, metric=metric)
        t.insert(0, "metric", metric)
        t.insert(0, "dataset", ds)
        boot_rows.append(t)

boot = pd.concat(boot_rows, ignore_index=True)
display(boot)
boot.to_csv(out_dir / "group_bootstrap_ci.csv", index=False)

bad = boot[~boot["estimable"].astype(bool)]
if len(bad):
    print()
    print("NOT ESTIMABLE at this resampling level - do not quote these intervals:")
    for _, r_ in bad.iterrows():
        print(f"  {r_['dataset']:8s} {r_['metric']:9s} {r_['level']:14s} "
              f"n_groups={r_['n_groups']:>5}  "
              f"all classes in {100*r_['frac_replicates_all_classes']:.0f}% of replicates")
else:
    print()
    print("every interval is estimable at its resampling level")
"""),
    ("code", """
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.2 * len(DATASETS), 3.8),
                         squeeze=False)
for ax, ds in zip(axes[0], DATASETS):
    sub = boot[(boot.dataset == ds) & (boot.metric == "f1_macro")]
    ypos = np.arange(len(sub))
    # Clip the error bars at zero.  A negative arm is a real diagnostic - the
    # point estimate lies outside its interval - but matplotlib raises on it,
    # and the first run of this notebook died here before the paired tests ran.
    lo = np.clip(sub["point"] - sub["ci_low"], 0, None)
    hi = np.clip(sub["ci_high"] - sub["point"], 0, None)
    ax.errorbar(sub["point"], ypos, xerr=[lo, hi], fmt="o", capsize=5,
                color="#08519c")
    ticks = []
    for lvl, n, est in zip(sub["level"], sub["n_groups"], sub["estimable"]):
        ticks.append(f"{lvl} (n={n})" + ("" if est else "  [not estimable]"))
    ax.set_yticks(ypos)
    ax.set_yticklabels(ticks)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 (%)")
    ax.set_title(f"{ds}: 95% CI by resampling unit")
plt.tight_layout()
fig.savefig(out_dir / "group_bootstrap_ci.png", dpi=200, bbox_inches="tight")
plt.show()
"""),
    ("md", """
## 3. Paired comparison over independent acquisitions

The direct replacement for the Friedman test. The same six balancing strategies are
compared, but each block is now a **held-out acquisition** - a whole Tomasov recording or
a whole Cao acquisition date - trained on all the others. Those blocks share no samples,
no sensor deployment and no acquisition session, so a paired Wilcoxon over them is a test
of the thing we actually care about.

This is also a much harder evaluation than the within-acquisition split, so the absolute
numbers will be lower. That is the point: the ranking is what is being tested, not the
level.

**Two rules make the metric mean something.** Classes confined to a single acquisition are
dropped, because holding that acquisition out removes the class from training altogether -
five of the nine Tomasov classes are in exactly that position, which is why notebook 08
restricted its leave-one-recording-out study to the four classes with two or more
recordings. And macro-F1 is averaged over the classes the held-out unit actually contains,
because `score_all` averages over the union of true and predicted labels, so a class the
model predicts but which is absent from that acquisition contributes a zero.

**Accuracy is the primary paired metric here**, since it is unaffected by class absence.
Macro-F1 over present classes is reported alongside it, and the uncorrected
`f1_macro_union` is kept so the correction is auditable.
"""),
    ("code", """
loo_path = out_dir / "leave_one_acquisition_out_scores.csv"
loo_cached = pd.read_csv(loo_path) if (REUSE_CACHED and loo_path.exists()) else None
if loo_cached is not None:
    print(f"reusing {loo_path.name} ({len(loo_cached)} rows). "
          f"Set REUSE_CACHED = False to recompute.")

loo_all, pair_all = [], []
for ds in DATASETS:
    d = DATA[ds]
    col = GROUP_COLS[ds]["loo"]
    idx = R.subsample_index(d["y"], LOO_PER_CLASS, C.SEED)
    X, y = d["X"][idx], d["y"][idx]
    groups = d["manifest"].iloc[idx][col].to_numpy()
    print(f"=== {ds}: leave-one-{col}-out over {len(np.unique(groups))} units ===")
    if loo_cached is not None and (loo_cached["dataset"] == ds).any():
        sc = loo_cached[loo_cached["dataset"] == ds].reset_index(drop=True)
        print(f"  reused {len(sc)} cached rows")
    else:
        t0 = time.perf_counter()
        sc = R.leave_one_group_out_scores(X, y, groups, PAIRED_CONFIGS, seed=C.SEED)
        sc["dataset"] = ds
        sc["unit"] = col
        print(f"  {time.perf_counter()-t0:.0f}s")
    loo_all.append(sc)

    for metric in ("accuracy", "f1_macro"):
        pc = R.paired_group_comparison(sc, metric=metric)
        pc.insert(0, "metric", metric)
        pc.insert(0, "dataset", ds)
        pair_all.append(pc)

loo_all = pd.concat(loo_all, ignore_index=True)
pair_all = pd.concat(pair_all, ignore_index=True)
loo_all.to_csv(out_dir / "leave_one_acquisition_out_scores.csv", index=False)
pair_all.to_csv(out_dir / "paired_wilcoxon_holm.csv", index=False)

print("mean accuracy per condition, averaged over held-out acquisitions:")
display(loo_all.groupby(["dataset", "condition"])["accuracy"]
        .agg(["count", "mean", "std"]).round(2))
print("mean macro-F1 (over classes present in each held-out unit):")
display(loo_all.groupby(["dataset", "condition"])["f1_macro"]
        .agg(["count", "mean", "std"]).round(2))
print("pairwise tests (Holm-corrected):")
display(pair_all.round(4))
"""),
    ("code", """
flag = (pair_all["significant_005"].fillna(False)
        if "significant_005" in pair_all.columns
        else pd.Series(False, index=pair_all.index))
sig = pair_all[flag.astype(bool)]
print(f"{len(sig)} of {len(pair_all)} pairwise comparisons survive Holm correction")
if len(sig):
    display(sig[["dataset", "condition_a", "condition_b", "n_units",
                 "mean_diff", "p_value", "p_holm"]].round(4))
else:
    print("No pair separates once the blocks are independent acquisitions.")
    print("That is a defensible result and the paper should state it plainly:")
    print("the choice of balancing strategy does not survive a test whose units")
    print("are separate acquisitions, and the original Friedman significance was")
    print("an artefact of blocking on configurations of a single partition.")

verdict = {
    "n_pairs": int(len(pair_all)),
    "n_significant_holm": int(len(sig)),
    "repeat_sd_f1_macro": {
        ds: float(np.nan_to_num(
            R.repeat_summary(rep_all[rep_all.dataset == ds])["sd"].max()))
        for ds in DATASETS},
    "bootstrap_widening": {
        ds: {row["level"]: float(2 * row["half_width"])
             for _, row in boot[(boot.dataset == ds) &
                                (boot.metric == "f1_macro")].iterrows()}
        for ds in DATASETS},
    "bootstrap_not_estimable": [
        f"{r_['dataset']}/{r_['metric']}/{r_['level']}"
        for _, r_ in boot[~boot["estimable"].astype(bool)].iterrows()],
    "note": ("Friedman/Nemenyi over balancing strategies is retired; see the "
             "three tests in this notebook."),
}
(out_dir / "statistics_verdict.json").write_text(json.dumps(verdict, indent=2),
                                                 encoding="utf-8")
print(json.dumps(verdict, indent=2))
"""),
    ("md", """
## 3b. Reconciling the 98.90 to 57.82 drop that R2C5 cites

R2C5 quotes the Cao leave-one-date-out macro-F1 falling from 98.90 to 57.82 as evidence of
extreme acquisition-to-acquisition variation. That number comes from notebook 08, and it
needs a correction before it can carry an argument.

Seven of the ten held-out dates contain only two to four of the six classes. Because
macro-F1 was averaged over the union of true and predicted labels, every class the model
predicted on a date where that class does not occur scored zero. Date 220113 is the clearest
case: 98.56% accuracy, 49.62 macro-F1.

The variation is real, but it is smaller than 57.82 suggests, and it is concentrated in two
dates rather than spread across all ten. The cell below recomputes both conventions from the
stored notebook 08 output so the manuscript can quote the corrected figure and say what
changed.
"""),
    ("code", """
lodo_path = C.RESULTS_DIR / "shared" / "08_crossdataset" / "cao_leave_one_date_out.csv"
if lodo_path.exists():
    lodo = pd.read_csv(lodo_path)
    display(lodo.round(2))
    print(f"dates evaluated                       : {len(lodo)}")
    print(f"dates holding fewer than 6 classes    : "
          f"{int((lodo['n_classes_in_test'] < 6).sum())}")
    print(f"mean macro-F1 as published (union)    : {lodo['f1_macro'].mean():.2f}")
    print(f"mean accuracy                         : {lodo['accuracy'].mean():.2f}")
    print(f"worst two dates by accuracy           : "
          f"{', '.join(lodo.nsmallest(2, 'accuracy')['held_out_date'].astype(str))}")
    card = json.loads((C.results_dir("cao", "07_final", FEATURE_SET) / "model_card.json")
                      .read_text(encoding="utf-8"))
    within = round(float(card["test"]["accuracy"]), 2)     # notebook 07's headline
    print()
    print(f"within-dataset accuracy               : {within:.2f}")
    print(f"drop in accuracy across dates          : "
          f"{within - lodo['accuracy'].mean():.2f} pp")
    lodo_fix = {
        "n_dates": int(len(lodo)),
        "n_dates_missing_classes": int((lodo["n_classes_in_test"] < 6).sum()),
        "published_macro_f1_union": round(float(lodo["f1_macro"].mean()), 2),
        "mean_accuracy": round(float(lodo["accuracy"].mean()), 2),
        "within_dataset_accuracy": within,
        "accuracy_drop_pp": round(within - float(lodo["accuracy"].mean()), 2),
        "note": ("macro-F1 was averaged over union(y_true, y_pred); on dates holding "
                 "a subset of the classes this scores absent classes as zero. "
                 "Accuracy is unaffected and is the figure the manuscript should lead "
                 "with, with the two genuinely failing dates named."),
    }
    (out_dir / "cao_lodo_reconciliation.json").write_text(
        json.dumps(lodo_fix, indent=2), encoding="utf-8")
    print()
    print("wrote", out_dir / "cao_lodo_reconciliation.json")
else:
    print("notebook 08 output not found at", lodo_path)
    print("Run notebook 08 first, or copy cao_leave_one_date_out.csv into place.")
"""),
    ("md", """
## 4. What the paper now says

Replace Table 14 and its critical-difference diagram with three items:

0. **Correct the acquisition-variation figure.** The 98.90 to 57.82 drop quoted in the
   manuscript, and repeated by the reviewer, is partly an artefact of averaging macro-F1
   over classes absent from the held-out date. Quote accuracy instead, name the two dates
   that genuinely fail, and say what was corrected - from `cao_lodo_reconciliation.json`.
1. **Partition variance.** Report mean and standard deviation of macro-F1 across the five
   repeated partitions for the headline configuration, from
   `repeated_splits_summary.csv`. Every headline number in the paper should carry that
   standard deviation.
2. **Honest intervals.** Report the recording-level (Tomasov) and date-level (Cao) 95%
   bootstrap intervals from `group_bootstrap_ci.csv`, and state in one sentence how much
   narrower the naive window-level interval was. That single comparison makes the
   dependence argument concretely rather than rhetorically.
3. **Paired tests where they are valid.** Report the Holm-corrected Wilcoxon results from
   `paired_wilcoxon_holm.csv`, blocked on held-out acquisitions.

State explicitly that the Friedman/Nemenyi analysis has been withdrawn, and why. A
withdrawn test that is explained is a strength; a defended one that the reviewer has
already seen through is not.

If test 3 finds no significant pair, say so. The claim of the paper is about the leakage
protocol and the multi-domain feature set, not about which resampler wins - and a null
result on the resampler comparison costs the paper nothing.
"""),
]
