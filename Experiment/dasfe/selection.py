"""Ensemble feature selection: 8 methods + consensus voting.

**Leakage rule enforced here:** every selector receives training-fold data
only.  `fit_all` takes `X_train, y_train` and never sees validation or test
rows; the scaler used by the scale-sensitive methods is fitted inside this
function on the training fold as well.  Notebook 05 additionally re-fits the
whole ensemble inside each cross-validation fold, so the reported CV numbers
carry no selection bias - the original submission fitted selection once on the
full training set, which biases CV estimates upward even when the test set is
untouched.

**Reproducibility rule:** the selected subsets must not depend on the machine.
Features whose values are decided by floating-point rounding
(`config.UNSTABLE_FEATURES`, empty since the extractors became exact) are never
offered to a selector, and no selection cut is ever decided by the order a sort
happens to leave tied scores in:
methods that rank in tied blocks (Boruta, RFE) order each block by their own
evidence, mRMR ranks every feature, and any exact tie that remains goes to the
feature the other methods rank higher, then to the lower column index.

Methods: Variance, one-vs-rest Pearson, Mutual Information, ReliefF, mRMR,
RFE (balanced logistic regression), Boruta (random forest), SHAP (TreeExplainer
on a random forest).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C

warnings.filterwarnings("ignore", category=UserWarning)


def _stratified_subsample(X, y, n_max, seed=C.SEED):
    """Class-proportional subsample for the selectors that scale badly."""
    if n_max is None or len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        take = max(1, int(round(n_max * cnt / len(y))))
        pool = np.flatnonzero(y == cls)
        idx.append(rng.choice(pool, size=min(take, pool.size), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


def _unit_rank(v: np.ndarray) -> np.ndarray:
    """Order-preserving map of `v` into (0, 1).

    Added to an integer rank it orders the features inside each rank group
    without moving any feature into a neighbouring group.
    """
    return rankdata(v, method="average") / (len(v) + 1)


# ==========================================================================
# Individual selectors - each returns a score per feature (higher = better)
# ==========================================================================
def score_variance(X, y, **kw):
    return X.var(axis=0)


def score_pearson(X, y, **kw):
    """One-vs-rest mean absolute Pearson correlation across classes."""
    Xc = X - X.mean(axis=0)
    sx = Xc.std(axis=0) + 1e-12
    scores = np.zeros(X.shape[1])
    for cls in np.unique(y):
        t = (y == cls).astype(float)
        tc = t - t.mean()
        st = tc.std() + 1e-12
        scores += np.abs((Xc * tc[:, None]).mean(axis=0) / (sx * st))
    return scores / len(np.unique(y))


def score_mutual_info(X, y, seed=C.SEED, n_max=C.FS_SUBSAMPLE, **kw):
    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    return mutual_info_classif(Xs, ys, n_neighbors=3, random_state=seed)


def score_relieff(X, y, seed=C.SEED, n_max=6000, **kw):
    from skrebate import ReliefF

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    sel = ReliefF(n_neighbors=10, n_features_to_select=Xs.shape[1], n_jobs=C.N_JOBS)
    sel.fit(Xs.astype(np.float64), ys)
    return np.asarray(sel.feature_importances_, dtype=float)


def score_mrmr(X, y, k=None, n_max=C.FS_SUBSAMPLE, seed=C.SEED,
               mrmr_k=C.MRMR_K, **kw):
    """Maximum relevance, minimum redundancy.

    `mrmr_classif` defaults: F-statistic relevance, Pearson-correlation
    redundancy.  Every feature is ranked (`mrmr_k=None`): a partial ranking
    leaves the unranked features tied at score 0, and the top-half cut would
    then pick among them arbitrarily.  Ranking all features costs about 50%
    more mRMR time than ranking 400.
    """
    from mrmr import mrmr_classif

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    k = Xs.shape[1] if mrmr_k is None else min(mrmr_k, Xs.shape[1])
    df = pd.DataFrame(Xs, columns=[str(i) for i in range(Xs.shape[1])])
    order = mrmr_classif(X=df, y=pd.Series(ys), K=k, show_progress=False)
    scores = np.zeros(X.shape[1])
    for rank, col in enumerate(order):
        scores[int(col)] = len(order) - rank        # first picked = highest score
    return scores


def score_rfe(X, y, n_max=C.FS_SUBSAMPLE, seed=C.SEED, step=0.1, **kw):
    """Recursive feature elimination with a balanced logistic regression.

    RFE drops a block of features per step and gives the whole block one rank.
    Inside each block, features are ordered by the importance RFE dropped them
    on - squared coefficients summed over classes, from the fit on the
    features still present at that step - so the ranking has no ties.

    The regression is solved to tol=1e-8 (about 2,300 / 5,800 L-BFGS iterations
    on Cao / Tomasov).  At the default 1e-4 the coefficients stop wherever the
    solver halts, and two machines whose features differ in the last bits
    eliminated 15 of the 483 kept features differently; at 1e-8, one.
    """
    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    Xz = StandardScaler().fit_transform(Xs)
    est = LogisticRegression(solver="lbfgs", max_iter=20000, tol=1e-8, class_weight="balanced",
                             n_jobs=C.N_JOBS)
    rank = RFE(est, n_features_to_select=1, step=step).fit(Xz, ys).ranking_
    within = np.zeros(rank.size)
    for r in np.unique(rank):
        block = rank == r
        if block.sum() > 1:
            alive = rank <= r                           # the features present at that step
            coef = np.atleast_2d(clone(est).fit(Xz[:, alive], ys).coef_)
            within[block] = (coef ** 2).sum(axis=0)[block[alive]]
    return -rank.astype(float) + _unit_rank(within)  # rank 1 is best -> highest score


def score_boruta(X, y, n_max=8000, seed=C.SEED, max_iter=60, **kw):
    """Boruta on a random forest.

    Boruta's ranking ties every feature it confirms at rank 1 (763 of 1,002 on
    Cao).  Inside each rank, features are ordered by their median importance
    over Boruta's iterations - the evidence its own tests are based on.
    """
    from boruta import BorutaPy

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    rf = RandomForestClassifier(n_estimators=800, class_weight="balanced",
                                max_depth=8, n_jobs=C.N_JOBS, random_state=seed)
    bor = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=seed, verbose=0)
    bor.fit(Xs.astype(np.float64), ys.astype(str))
    history = np.atleast_2d(np.asarray(bor.importance_history_, dtype=float))[1:]  # row 0 = zeros
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # columns that are all NaN
        median_imp = np.nanmedian(history, axis=0)
    median_imp = np.nan_to_num(median_imp, nan=-np.inf)
    return -bor.ranking_.astype(float) + _unit_rank(median_imp)


def score_shap(X, y, n_max=6000, seed=C.SEED, **kw):
    import shap

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                n_jobs=C.N_JOBS, random_state=seed)
    rf.fit(Xs, ys)
    expl = shap.TreeExplainer(rf)
    vals = expl.shap_values(Xs[: min(2000, len(Xs))], check_additivity=False)
    vals = np.asarray(vals)
    # shape is (n_samples, n_features, n_classes) or a list per class
    axes = tuple(a for a in range(vals.ndim) if a != vals.ndim - 2) if vals.ndim == 3 else 0
    return np.abs(vals).mean(axis=axes) if vals.ndim == 3 else np.abs(vals).mean(axis=0)


SCORERS = {
    "variance": score_variance,
    "pearson": score_pearson,
    "mutual_info": score_mutual_info,
    "relieff": score_relieff,
    "mrmr": score_mrmr,
    "rfe": score_rfe,
    "boruta": score_boruta,
    "shap": score_shap,
}


# ==========================================================================
# Ensemble
# ==========================================================================
def _order(scores: dict, ranks: dict, method: str) -> np.ndarray:
    """Column indices, best first, by a rule that is the same on every machine.

    Score first; an exact tie goes to the feature the other methods rank
    higher on average, then to the lower column index.  Ineligible features
    (score -inf) come last.
    """
    s = scores[method]
    others = [r for m, r in ranks.items() if m != method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ineligible columns are all NaN
        others_mean = np.nanmean(others, axis=0) if others else np.zeros(s.size)
    others_mean = np.nan_to_num(others_mean, nan=np.inf)
    return np.lexsort((np.arange(s.size), others_mean, -s))


def fit_all(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str],
            methods=C.FS_METHODS, top_frac: float = 0.5,
            seed: int = C.SEED, verbose: bool = True, fast: bool = False,
            exclude=C.UNSTABLE_FEATURES) -> dict:
    """Score every feature with every method, using training data only.

    Features named in `exclude` (by default `config.UNSTABLE_FEATURES`) are not
    offered to any selector: they get no score, no rank and no vote.  Each
    method keeps the top `top_frac` of the eligible features, so all methods
    vote on a comparably sized subset and the consensus is not dominated by
    whichever method happens to be least selective.  Masks, votes, scores and
    the table are all indexed by the full feature list.

    `fast=True` shrinks the subsample budgets (see `config.FS_FAST_KW`) without
    changing which methods run.  It is used for the in-fold refits, where the
    ensemble is evaluated `n_folds` times.
    """
    names = list(feature_names)
    n_all = X_train.shape[1]
    excluded = set(exclude or ())
    eligible = np.array([n not in excluded for n in names])
    cols = np.flatnonzero(eligible)
    X_elig = X_train[:, cols]
    n_feat = cols.size
    k = max(1, int(round(top_frac * n_feat)))
    scores, timings = {}, {}

    for name in methods:
        t0 = time.perf_counter()
        kw = dict(C.FS_FAST_KW.get(name, {})) if fast else {}
        try:
            s = np.asarray(SCORERS[name](X_elig, y_train, seed=seed, **kw), dtype=float)
            s = np.nan_to_num(s, nan=-np.inf)
        except Exception as exc:                      # keep the ensemble alive
            if verbose:
                print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
            continue
        if s.size != n_feat:
            if verbose:
                print(f"  [{name}] wrong score length {s.size} != {n_feat}, skipped")
            continue
        full = np.full(n_all, -np.inf)
        full[cols] = s
        scores[name] = full
        timings[name] = time.perf_counter() - t0
        if verbose:
            print(f"  [{name}] scored {n_feat} eligible features in {timings[name]:.1f}s")

    # Ranks among the eligible features (1 = best, ties share the average rank).
    ranks = {}
    for m, s in scores.items():
        r = np.full(n_all, np.nan)
        r[cols] = pd.Series(-s[cols]).rank(method="average").to_numpy()
        ranks[m] = r
    selected = {}
    for m in scores:
        mask = np.zeros(n_all, dtype=bool)
        mask[_order(scores, ranks, m)[:k]] = True
        selected[m] = mask
    if verbose and scores:
        held = f" ({n_all - n_feat} excluded features not offered)" if n_feat < n_all else ""
        print(f"  each method keeps {k} of {n_feat} eligible features{held}")

    votes = np.sum(list(selected.values()), axis=0) if selected else np.zeros(n_all, int)
    table = pd.DataFrame({"feature": names, "feature_index": np.arange(n_all),
                          "eligible": eligible, "votes": votes})
    for m, s in scores.items():
        table[f"score_{m}"] = np.where(eligible, s, np.nan)
        table[f"rank_{m}"] = ranks[m]
    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    table["avg_rank"] = table[rank_cols].mean(axis=1)
    table["selection_pct"] = 100 * table["votes"] / max(len(selected), 1)
    table = table.sort_values(["votes", "avg_rank", "feature_index"],
                              ascending=[False, True, True], kind="stable")

    return {
        "scores": scores,
        "ranks": ranks,
        "selected": selected,
        "votes": votes,
        "eligible": eligible,
        "table": table.reset_index(drop=True),
        "timings": timings,
        "n_methods": len(selected),
        "top_k": k,
    }


def top_k(result: dict, method: str, k: int) -> np.ndarray | None:
    """Sorted column indices of `method`'s `k` best eligible features.

    Uses the same deterministic rule as `fit_all`'s own cut; None if the
    method did not run.
    """
    if method not in result["scores"]:
        return None
    order = _order(result["scores"], result["ranks"], method)
    order = order[result["eligible"][order]]
    return np.sort(order[:k])


def consensus_mask(result: dict, threshold: int | None = None) -> np.ndarray:
    """Features chosen by at least `threshold` methods (default: all of them)."""
    threshold = threshold if threshold is not None else result["n_methods"]
    return result["votes"] >= threshold


def jaccard_matrix(result: dict) -> pd.DataFrame:
    """Pairwise agreement between selectors (Figure 6)."""
    methods = list(result["selected"])
    M = np.zeros((len(methods), len(methods)))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            A, B = result["selected"][a], result["selected"][b]
            union = (A | B).sum()
            M[i, j] = (A & B).sum() / union if union else 1.0
    return pd.DataFrame(M, index=methods, columns=methods)


def save(result: dict, out_dir: Path, tag: str = "multi") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(out_dir / f"fs_table_{tag}.csv", index=False)
    jaccard_matrix(result).to_csv(out_dir / f"fs_jaccard_{tag}.csv")
    np.save(out_dir / f"fs_votes_{tag}.npy", result["votes"])
    np.savez(out_dir / f"fs_masks_{tag}.npz", **{m: v for m, v in result["selected"].items()})
    with open(out_dir / f"fs_timings_{tag}.json", "w") as fh:
        json.dump(result["timings"], fh, indent=1)
