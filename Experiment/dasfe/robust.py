"""Repeated group-aware splits, group-level bootstrap, and paired testing.

Written in response to reviewer comment R2C5, which objects that a Friedman test
over six balancing strategies on a single partition treats non-independent
conditions as experimental blocks.  Those conditions share samples, share the
partition, and share almost every source of variance, so the resulting p-value
overstates the evidence.

Three replacements are provided, in increasing order of how independent the
compared units actually are:

1. `run_repeated` - re-partition the same population with several seeds and
   report the spread of test performance across partitions.  This captures the
   partition variance that a single split cannot expose.
2. `group_bootstrap_ci` - resample whole recordings or acquisition dates with
   replacement from the test set.  Resampling groups rather than windows respects
   the dependence structure and yields an honest interval.
3. `leave_one_group_out_scores` + `paired_group_comparison` - compare conditions
   on genuinely independent acquisition units with a Wilcoxon signed-rank test on
   the paired differences, Holm-corrected across the pairwise family.

The third is the strongest, because its blocking units are separate acquisitions
rather than separate configurations of one acquisition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config as C
from . import evaluate as EV
from . import splits as S


# ==========================================================================
# Shared helpers
# ==========================================================================
def subsample_index(y, per_class: int | None, seed: int = C.SEED) -> np.ndarray:
    """Stratified row subsample, or every row when `per_class` is None."""
    y = np.asarray(y)
    if per_class is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(y):
        idx = np.flatnonzero(y == lab)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def _label(model: str, strategy: str) -> str:
    return f"{model} / {strategy}"


# ==========================================================================
# 1. Repeated group-aware partitions
# ==========================================================================
def repeated_group_splits(
    manifest: pd.DataFrame,
    dataset: str,
    n_repeats: int = 5,
    base_seed: int = C.SEED,
) -> list[pd.Series]:
    """Return `n_repeats` split assignments over the same population.

    The population is held fixed - it is the set of rows for which features
    already exist - and only the assignment varies, which isolates partition
    variance from population variance.

    For Tomasov the guard band is re-applied for each seed, so rows adjacent to a
    boundary are dropped or kept depending on the assignment; the returned Series
    carries NaN for rows dropped under that seed.
    """
    out = []
    for k in range(n_repeats):
        seed = base_seed + 1000 * k
        if dataset == "tomasov":
            assigned = S.blocked_split_by_recording(
                manifest, recording_col="recording_id", block_col="block",
                n_segments=C.N_SEGMENTS, fractions=C.SPLIT_FRACTIONS, seed=seed,
            )
            tmp = manifest.copy()
            tmp["split"] = assigned["split"].to_numpy()
            keep = S.apply_temporal_guard(
                tmp, split_col="split", recording_col="recording_id",
                block_col="block")
            s = assigned["split"].where(keep.to_numpy(), other=None)
        else:
            s = S.grouped_three_way_split(
                manifest, group_key=C.DATASETS[dataset].group_key, seed=seed)
        s = pd.Series(np.asarray(s, dtype=object), index=manifest.index,
                      name=f"split_seed{seed}")
        out.append(s)
    return out


def run_repeated(
    X: np.ndarray,
    y: np.ndarray,
    manifest: pd.DataFrame,
    dataset: str,
    configs,
    n_repeats: int = 5,
    base_seed: int = C.SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit each (model, strategy) on every repeated partition.

    `configs` is an iterable of (model_name, strategy) pairs.  One row is
    returned per (config, repeat).
    """
    assignments = repeated_group_splits(manifest, dataset, n_repeats, base_seed)
    rows = []
    for k, s in enumerate(assignments):
        sp = s.to_numpy()
        tr = sp == "train"
        te = sp == "test"
        if te.sum() < 50 or tr.sum() < 100:
            if verbose:
                print(f"  repeat {k}: skipped (train {tr.sum()}, test {te.sum()})")
            continue
        for model, strategy in configs:
            try:
                r = EV.fit_predict(model, strategy, X[tr], y[tr], X[te], y[te],
                                   base_seed + 1000 * k)
            except Exception as exc:
                if verbose:
                    print(f"  repeat {k} {_label(model, strategy)}: "
                          f"FAILED {type(exc).__name__}: {exc}")
                continue
            m = r["metrics"]
            rows.append({
                "repeat": k, "seed": base_seed + 1000 * k,
                "model": model, "strategy": strategy,
                "condition": _label(model, strategy),
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "accuracy": 100 * m["accuracy"],
                "balanced_accuracy": 100 * m["balanced_accuracy"],
                "f1_macro": 100 * m["f1_macro"],
            })
            if verbose:
                print(f"  repeat {k}  {_label(model, strategy):34s} "
                      f"F1 {100*m['f1_macro']:6.2f}  (test {int(te.sum()):,})",
                      flush=True)
    return pd.DataFrame(rows)


def repeat_summary(rep: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Mean, standard deviation and range of a metric across repeated splits."""
    g = rep.groupby("condition")[metric]
    out = pd.DataFrame({
        "n_repeats": g.size(),
        "mean": g.mean(),
        "sd": g.std(ddof=1),
        "min": g.min(),
        "max": g.max(),
    })
    out["range"] = out["max"] - out["min"]
    return out.sort_values("mean", ascending=False).round(3)


# ==========================================================================
# 2. Group-level bootstrap
# ==========================================================================
def group_bootstrap_ci(
    y_true,
    y_pred,
    groups,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = C.SEED,
    metric: str = "f1_macro",
) -> dict:
    """Percentile bootstrap over whole groups rather than individual windows.

    Resampling windows would treat tens of thousands of highly dependent
    observations as independent and produce an interval far too narrow.
    Resampling recordings or acquisition dates respects the dependence, and is
    the interval the paper reports.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    all_labels = np.unique(y_true)

    def score(idx):
        """Return (metric, number of classes the replicate contains)."""
        if idx.size == 0:
            return np.nan, 0
        if metric == "accuracy":
            return 100 * float((y_true[idx] == y_pred[idx]).mean()), all_labels.size
        # Macro-F1 is averaged over the classes the replicate actually contains.
        # Holding the label set at every class scores an absent one as zero,
        # which on Tomasov - where five of nine classes come from a single
        # recording each - drags the replicate distribution so far below the
        # point estimate that the point falls outside its own interval. The
        # estimand becomes "macro-F1 over the classes present", and the
        # `frac_replicates_all_classes` field below reports how often that
        # differs from the full set.
        present = np.unique(y_true[idx])
        return (100 * f1_score(y_true[idx], y_pred[idx], average="macro",
                               labels=present, zero_division=0), present.size)

    point = score(np.arange(len(y_true)))[0]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n_cls = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        boots[b], n_cls[b] = score(idx)
    ok = np.isfinite(boots)
    boots, n_cls = boots[ok], n_cls[ok]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    frac_all = float((n_cls == all_labels.size).mean()) if n_cls.size else float("nan")
    # An interval that excludes its own point estimate, or one whose replicates
    # routinely lose whole classes, is not an interval on the quantity we mean.
    estimable = bool(lo <= point <= hi and frac_all >= 0.9)
    return {
        "metric": metric,
        "point": round(float(point), 3),
        "ci_low": round(float(lo), 3),
        "ci_high": round(float(hi), 3),
        "half_width": round(float(hi - lo) / 2, 3),
        "n_groups": int(uniq.size),
        "n_boot": int(boots.size),
        "mean_classes_present": round(float(n_cls.mean()), 2) if n_cls.size else None,
        "n_classes_total": int(all_labels.size),
        "frac_replicates_all_classes": round(frac_all, 3),
        "estimable": estimable,
    }


def bootstrap_by_level(y_true, y_pred, level_groups: dict, **kw) -> pd.DataFrame:
    """Run `group_bootstrap_ci` at several resampling levels and tabulate.

    `level_groups` maps a level name to the group array, e.g.
    {"window (naive)": arange(n), "zone": zones, "recording": recordings}.
    Widening intervals as the unit grows coarser is the point of the table: it
    shows how much a window-level interval understates the true uncertainty.
    """
    rows = []
    for name, g in level_groups.items():
        r = group_bootstrap_ci(y_true, y_pred, g, **kw)
        r["level"] = name
        rows.append(r)
    cols = ["level", "n_groups", "point", "ci_low", "ci_high", "half_width",
            "mean_classes_present", "n_classes_total",
            "frac_replicates_all_classes", "estimable", "n_boot"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


# ==========================================================================
# 3. Paired comparison across independent acquisition units
# ==========================================================================
def multigroup_classes(y, groups, min_groups: int = 2) -> np.ndarray:
    """Classes that appear in at least `min_groups` distinct acquisition units.

    A class confined to one recording cannot be evaluated by leaving that
    recording out: holding it out removes the class from training entirely, and
    holding out any other recording removes it from test. Five of the nine
    Tomasov classes are in exactly this position.
    """
    t = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    n = t.groupby("y")["g"].nunique()
    return np.array(sorted(n[n >= min_groups].index))


def leave_one_group_out_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    configs,
    seed: int = C.SEED,
    min_test: int = 30,
    min_classes: int = 1,
    restrict_to_multigroup: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every condition on every held-out acquisition unit.

    Each held-out group is one genuinely independent experimental unit, which is
    what a paired test over conditions should be blocked on.

    Two rules keep the metric well defined:

    * with `restrict_to_multigroup`, classes confined to a single unit are
      dropped, because leaving that unit out makes the class untrainable;
    * macro-F1 is averaged over the classes the held-out unit actually contains.
      `EV.score_all` averages over union(y_true, y_pred), so a class the model
      predicts but which is absent from the held-out acquisition contributes a
      zero. That is what turns 98.56% accuracy into 49.62 macro-F1 on Cao date
      220113, and it is a property of the metric, not of generalisation.

    A fold holding a single class gets `f1_macro = NaN`, because a macro average
    over one class is not a quantity. This is the normal case on Tomasov, where
    every recording contains exactly one event type, so the Tomasov paired test
    necessarily runs on accuracy - which for a one-class fold is recall on the
    held-out class. Notebook 08 made the same choice for the same reason.

    Accuracy is unaffected by class absence and is the safer paired metric here.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    groups = np.asarray(groups)

    if restrict_to_multigroup:
        keep_cls = multigroup_classes(y, groups, 2)
        mask = np.isin(y, keep_cls)
        dropped = sorted(set(np.unique(y)) - set(keep_cls))
        if verbose and dropped:
            print(f"  dropping {len(dropped)} single-unit class(es): {dropped}")
        X, y, groups = X[mask], y[mask], groups[mask]

    rows = []
    for g in sorted(pd.unique(groups)):
        te = groups == g
        tr = ~te
        present = np.unique(y[te])
        if te.sum() < min_test or len(np.unique(y[tr])) < 2 or len(present) < min_classes:
            if verbose:
                print(f"  {str(g)[:26]:28s} skipped "
                      f"(n={int(te.sum())}, {len(present)} class(es) in test)")
            continue
        got = {}
        for model, strategy in configs:
            try:
                r = EV.fit_predict(model, strategy, X[tr], y[tr], X[te], y[te], seed)
            except Exception as exc:
                if verbose:
                    print(f"  {str(g)[:26]:28s} {_label(model, strategy):30s} "
                          f"FAILED {type(exc).__name__}")
                continue
            m = r["metrics"]
            cond = _label(model, strategy)
            # A macro average over one class is not a quantity; leave it NaN so
            # it cannot be averaged into a table or tested.
            f1_present = (
                100 * f1_score(y[te], r["y_pred"], average="macro",
                               labels=present, zero_division=0)
                if len(present) >= 2 else np.nan
            )
            rows.append({"held_out": str(g), "condition": cond,
                         "model": model, "strategy": strategy,
                         "n_test": int(te.sum()),
                         "n_classes_in_test": int(len(present)),
                         "accuracy": 100 * m["accuracy"],
                         "balanced_accuracy": 100 * m["balanced_accuracy"],
                         "f1_macro": f1_present,
                         "f1_macro_union": 100 * m["f1_macro"]})
            got[cond] = 100 * m["accuracy"]
        if verbose and got:
            print(f"  {str(g)[:26]:28s} n={int(te.sum()):6,} "
                  f"k={len(present)}  acc " +
                  "  ".join(f"{k.split(' / ')[-1]}={v:5.1f}" for k, v in got.items()),
                  flush=True)

    cols = ["held_out", "condition", "model", "strategy", "n_test",
            "n_classes_in_test", "accuracy", "balanced_accuracy",
            "f1_macro", "f1_macro_union"]
    if not rows:
        # A legitimate outcome when every acquisition is too small or holds a
        # class that exists nowhere else.  Return the schema so downstream
        # concatenation and pivoting stay well defined.
        if verbose:
            print("  no acquisition unit was scorable under the current filters")
        return pd.DataFrame({c: pd.Series(dtype="object" if c in
                                          ("held_out", "condition", "model",
                                           "strategy") else "float64")
                             for c in cols})
    return pd.DataFrame(rows)[cols]


def paired_group_comparison(scores: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Wilcoxon signed-rank over held-out acquisition units, for every pair.

    Units are separate recordings or acquisition dates, so the paired differences
    are between genuinely independent experiments rather than between
    configurations that share a partition.  Holm correction is applied across the
    pairwise family.
    """
    from scipy.stats import wilcoxon

    cols = ["condition_a", "condition_b", "n_units", "mean_diff", "median_diff",
            "wins_a", "wins_b", "wilcoxon_stat", "p_value", "p_holm",
            "significant_005"]
    empty = pd.DataFrame({c: pd.Series(dtype="object" if c.startswith("condition")
                                       else "float64") for c in cols})
    if not len(scores) or "held_out" not in scores.columns:
        return empty
    if metric not in scores.columns or scores[metric].notna().sum() == 0:
        # Normal on Tomasov for macro-F1: every recording holds one class, so the
        # macro average is undefined and the paired test runs on accuracy alone.
        return empty

    wide = scores.pivot(index="held_out", columns="condition", values=metric).dropna()
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        return empty
    conds = list(wide.columns)
    rows = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b = conds[i], conds[j]
            d = wide[a] - wide[b]
            if len(d) < 3 or np.allclose(d, 0):
                stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = wilcoxon(wide[a], wide[b])
                except ValueError:
                    stat, p = np.nan, np.nan
            rows.append({
                "condition_a": a, "condition_b": b, "n_units": int(len(d)),
                "mean_diff": round(float(d.mean()), 3),
                "median_diff": round(float(d.median()), 3),
                "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
                "wilcoxon_stat": stat, "p_value": p,
            })
    out = pd.DataFrame(rows)
    if len(out) and out["p_value"].notna().any():
        m = int(out["p_value"].notna().sum())
        order = out["p_value"].rank(method="first")
        out["p_holm"] = (out["p_value"] * (m - order + 1)).clip(upper=1.0)
        out["significant_005"] = out["p_holm"] < 0.05
    return out.sort_values("p_value", na_position="last").reset_index(drop=True)


# ==========================================================================
# 4. Protocol arms (R2C3)
# ==========================================================================
def permute_split_within_class(labels, split, seed: int = C.SEED) -> np.ndarray:
    """Shuffle split tags among the rows of each class.

    The result has exactly the same class-by-split counts as `split`, over
    exactly the same rows.  The only thing that changes is *which* rows are held
    out, so a contrast between the two isolates contiguity - that is, leakage -
    with population, training-set size and class balance all held fixed.  This is
    the cleanest available operationalisation of "only the protocol changed".
    """
    labels = np.asarray(labels)
    split = np.asarray(split, dtype=object)
    out = split.copy()
    rng = np.random.default_rng(seed)
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        out[idx] = rng.permutation(split[idx])
    return out


def class_split_fractions(labels, split) -> pd.DataFrame:
    """Per-class share of each split tag, for reproducing a partition's shape."""
    df = pd.DataFrame({"label": np.asarray(labels), "split": np.asarray(split)})
    return pd.crosstab(df["label"], df["split"], normalize="index")


def stratified_split_like(labels, fractions: pd.DataFrame, seed: int = C.SEED) -> np.ndarray:
    """Random assignment reproducing a reference partition's per-class shape.

    Used when the population itself differs from the reference - the stride-256
    arm, for instance - so a permutation is impossible but the shape can still be
    matched, keeping training-set size out of the comparison.
    """
    labels = np.asarray(labels)
    tags = list(fractions.columns)
    default = fractions.mean(axis=0).to_numpy(dtype=float)
    out = np.empty(len(labels), dtype=object)
    rng = np.random.default_rng(seed)
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        rng.shuffle(idx)
        fr = (fractions.loc[lab, tags].to_numpy(dtype=float)
              if lab in fractions.index else default)
        counts = np.floor(fr * len(idx)).astype(int)
        counts[int(np.argmax(fr))] += len(idx) - counts.sum()
        start = 0
        for tag, n in zip(tags, counts):
            out[idx[start:start + n]] = tag
            start += n
    return out


def jaccard(a, b) -> float:
    """Jaccard overlap of two feature-name collections."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0
