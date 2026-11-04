"""Find the features whose values are set by floating-point rounding.

Two independent tests, written to one table:

  perturbation   each sampled window is extracted twice - once as recorded and
                 once with its raw samples scaled by (1 + 1e-12 * noise), far
                 below any physical meaning.  A feature that changes is decided
                 by rounding, not by the signal.
  cross-machine  (optional) the same windows' stored features from two results
                 trees, e.g. the Jetson's and a re-run on another machine,
                 matched by sample_id.

A change counts when it exceeds TAU of the feature's own spread (standard
deviation across windows).  The tolerance must be relative to the spread, not
absolute: a feature that is constant by construction - the mean of a z-scored
window, or the near-zero DC bin of a de-meaned one - varies only by ~1e-17,
yet that rounding noise is all that tree splits and standardised regressions
see of it.  Constant features (spread exactly 0) are not flagged.  The result
backed `config.UNSTABLE_FEATURES` (36 features) until the extractors were made
exact; `_amplitude_audit.py` is the check that it now stays empty.

    python _feature_stability.py OUT.csv                       # perturbation test only
    python _feature_stability.py OUT.csv --compare A B         # plus cross-machine
    python _feature_stability.py OUT.csv --windows 500         # windows per dataset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dasfe import config as C, preprocess as pp
from dasfe.extract import DOMAIN_ORDER, extract_window

TAU, EPS = 1e-3, 1e-12


def _features(raw: np.ndarray, fs: float, ds: str, spec) -> np.ndarray:
    patch = pp.preprocess_patch(raw, fs)
    ch = pp.cao_reference_channel(raw) if ds == "cao" else pp.center_channel_index(spec.patch_channels)
    got = extract_window(patch[ch], patch, fs)
    return np.concatenate([np.asarray(got[d], dtype=np.float64) for d in DOMAIN_ORDER])


def _pair(ds: str, row: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One window's features as recorded and after a rounding-level nudge."""
    spec = C.DATASETS[ds]
    if ds == "cao":
        raw, fs = pp.load_cao_window(row["path"], spec.win_len), spec.fs
    else:
        with pp.TomasovReader(row["h5_path"]) as rd:
            raw = rd.read_patch(int(row["t0"]), int(row["locus"]), spec.win_len, spec.patch_channels)
            fs = rd.fs
    noise = np.random.default_rng(seed).standard_normal(raw.shape)
    return _features(raw, fs, ds, spec), _features(raw * (1 + EPS * noise), fs, ds, spec)


def _pct_changed(ref: np.ndarray, other: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """% of rows where |ref - other| exceeds TAU x the feature's spread."""
    with np.errstate(invalid="ignore"):
        hit = np.abs(ref - other) > TAU * spread
    hit &= ~(np.isnan(ref) & np.isnan(other))
    hit[:, spread == 0] = False                      # constant features are harmless
    return 100 * hit.mean(axis=0)


def perturbation(n_windows: int) -> pd.DataFrame:
    cols = {}
    for ds in ("cao", "tomasov"):
        mani = pd.read_parquet(C.RESULTS_DIR / ds / "01_splits" / "split_manifest.parquet")
        if "n_bytes" in mani:
            mani = mani[mani.n_bytes > 0]
        rows = mani.sample(n_windows, random_state=C.SEED).to_dict("records")
        pairs = Parallel(n_jobs=C.N_JOBS)(delayed(_pair)(ds, r, C.SEED + i) for i, r in enumerate(rows))
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        cols[f"perturb_pct_{ds}"] = _pct_changed(a, b, np.nanstd(a, axis=0))
        print(f"  {ds}: {n_windows} windows perturbed", flush=True)
    return pd.DataFrame(cols)


def cross_machine(a_root: Path, b_root: Path) -> pd.DataFrame:
    cols = {}
    for ds in ("cao", "tomasov"):
        fa, fb = a_root / ds / "04_fusion", b_root / ds / "04_fusion"
        ida = np.load(fa / "sample_ids.npy", allow_pickle=True)
        idb = np.load(fb / "sample_ids.npy", allow_pickle=True)
        pos = pd.Index(idb).get_indexer(ida)
        rows = np.flatnonzero(pos >= 0)
        Xa = np.load(fa / "X_multi.npy", mmap_mode="r")
        Xb = np.load(fb / "X_multi.npy", mmap_mode="r")
        spread = np.nanstd(np.asarray(Xa[rows], dtype=np.float64), axis=0)
        bad = np.zeros(Xa.shape[1])
        for i in range(0, rows.size, 20000):                    # chunked: 700 MB matrices
            ia = rows[i:i + 20000]
            bad += _pct_changed(np.asarray(Xa[ia], np.float64), np.asarray(Xb[pos[ia]], np.float64),
                                spread) * ia.size / 100
        cols[f"machine_pct_{ds}"] = 100 * bad / rows.size
        print(f"  {ds}: {rows.size:,} windows compared across machines", flush=True)
    return pd.DataFrame(cols)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0])
    n = int(argv[argv.index("--windows") + 1]) if "--windows" in argv else 500
    names = json.loads((C.RESULTS_DIR / "cao" / "04_fusion" / "feature_names.json").read_text())
    print(f"perturbation test, {n} windows per dataset ...", flush=True)
    table = pd.concat([pd.DataFrame({"feature": names}), perturbation(n)], axis=1)
    if "--compare" in argv:
        i = argv.index("--compare")
        print("cross-machine comparison ...", flush=True)
        table = pd.concat([table, cross_machine(Path(argv[i + 1]), Path(argv[i + 2]))], axis=1)
    table["max_pct"] = table.filter(regex="_pct_").max(axis=1)
    table = table.sort_values("max_pct", ascending=False, kind="stable")
    table.to_csv(out, index=False)
    flagged = table[table.max_pct > 0]
    print(f"\n{len(flagged)} features change by > {TAU:g} x their spread in at least one window "
          f"(written to {out}):")
    print(flagged.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
