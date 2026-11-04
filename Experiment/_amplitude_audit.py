"""Check that no feature sees per-channel amplitude or floating-point rounding.

Per-channel z-scoring makes every one of the 1,002 features invariant to a gain
applied to each raw channel, in exact arithmetic.  A feature that still moves
under such a gain is reading a numerical side channel - before the extractors
were made exact, 33 did, and 13 of them carried each channel's raw amplitude
(the recording fingerprint the normalisation is there to remove) through the
epsilon of the z-score.  Two tests on sampled windows, run through the same
path as the extraction (`preprocess_patch` -> `extract_window`, float32 as the
shards store it):

  gain           every raw channel multiplied by its own gain in [0.25, 4]
  perturbation   every raw sample scaled by (1 + 1e-12 * noise)

A feature moves when it changes by more than TAU of its own spread across the
sampled windows in more than 0.1% of them.  Exit status 1 if any feature moves.

    python _amplitude_audit.py                      # 8 windows per recording / class
    python _amplitude_audit.py --windows 20 --csv OUT.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dasfe import config as C, extract as EX, preprocess as pp                 # noqa: E402
from dasfe import features_freq, features_spatial, features_tf, features_time  # noqa: E402

TAU = 1e-3
NAMES = ([f"TIME:{n}" for n in features_time.feature_names()] +
         [f"FREQ:{n}" for n in features_freq.feature_names()] +
         [f"TF:{n}" for n in features_tf.feature_names()] +
         [f"SPAT:{n}" for n in features_spatial.feature_names()])


def features(raw: np.ndarray, fs: float, ref: int) -> np.ndarray:
    patch = pp.preprocess_patch(raw, fs)
    d = EX.extract_window(patch[ref], patch, fs)
    return np.concatenate([np.asarray(d[k], dtype=np.float32) for k in EX.DOMAIN_ORDER])


def windows(ds: str, n_per: int):
    """Raw patches, n_per per Tomasov recording or per Cao class."""
    spec = C.DATASETS[ds]
    m = pd.read_parquet(C.results_dir(ds, "01_splits") / "split_manifest.parquet")
    if ds == "tomasov":
        pick = (m.groupby("recording_id", group_keys=False)
                 .apply(lambda g: g.sample(min(n_per, len(g)), random_state=1), include_groups=False)
                 .sort_values(["h5_path", "t0"]))
        for path, g in pick.groupby("h5_path", sort=False):
            with pp.TomasovReader(path) as rd:
                for r in g.itertuples():
                    raw = rd.read_patch(int(r.t0), int(r.locus), spec.win_len, spec.patch_channels)
                    yield np.asarray(raw, dtype=np.float64), rd.fs, pp.center_channel_index(spec.patch_channels)
    else:
        m = m[m["n_bytes"] > 0]
        pick = m.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(n_per, len(g)), random_state=1), include_groups=False)
        for r in pick.itertuples():
            raw = np.asarray(pp.load_cao_window(r.path, spec.win_len), dtype=np.float64)
            # The reference channel is chosen once, from the untouched record, so
            # a gain cannot change which channel feeds the 1D descriptors.
            yield raw, spec.fs, pp.cao_reference_channel(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=8, help="per Tomasov recording / Cao class")
    ap.add_argument("--csv", type=Path, help="write the per-feature table here")
    args = ap.parse_args()

    rows, n_moved = [], 0
    for ds in ("tomasov", "cao"):
        rng = np.random.default_rng(7)
        base, gain, pert = [], [], []
        t0 = time.perf_counter()
        for raw, fs, ref in windows(ds, args.windows):
            g = np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=(raw.shape[0], 1)))
            base.append(features(raw, fs, ref))
            gain.append(features(raw * g, fs, ref))
            pert.append(features(raw * (1.0 + 1e-12 * rng.standard_normal(raw.shape)), fs, ref))
        base, gain, pert = (np.asarray(a, dtype=np.float64) for a in (base, gain, pert))
        spread = base.std(axis=0)
        tol = TAU * spread
        print(f"{ds}: {len(base)} windows in {time.perf_counter() - t0:.0f}s, "
              f"{int((spread == 0).sum())} of {len(NAMES)} features constant")
        for test, other in (("gain", gain), ("perturbation", pert)):
            pct = 100 * (np.abs(other - base) > tol).mean(axis=0)
            moved = [n for n, p in zip(NAMES, pct) if p > 0.1]
            n_moved += len(moved)
            print(f"  {test:12s}: {len(moved)} features move" + (f" - {', '.join(moved)}" if moved else ""))
            rows += [{"dataset": ds, "test": test, "feature": n, "pct_windows": p, "spread": s}
                     for n, p, s in zip(NAMES, pct, spread)]
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print("wrote", args.csv)
    return 1 if n_moved else 0


if __name__ == "__main__":
    sys.exit(main())
