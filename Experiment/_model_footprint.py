"""Model footprint of the selected LightGBM configurations.

Notebook 07 reports the selected model's metrics and timings but does not save
the fitted booster, so its size was never recorded.  This script refits each
selected configuration exactly as notebook 07 does (same features, split, model,
balancing strategy and seed) and records the size of the fitted model:

* trees, split nodes and leaves of the booster;
* stored parameters: one feature index and one threshold per split node plus
  one value per leaf (the tree analogue of a network's weight count);
* size of the saved model file (LightGBM text format) and of its gzip copy.

Test metrics are recomputed so the refit can be checked against the model card
of notebook 07.  Timings are not recorded: they depend on the machine.

Usage (from the repo root or Experiment/):
    python Experiment/_model_footprint.py [--out DIR] [tomasov|cao ...]
Reads DAS_RESULTS_DIR like every notebook; writes footprint.json/csv to --out
(default: <results>/shared/17_model_footprint).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from dasfe import config as C, evaluate as EV, fusion as F


def booster_stats(booster) -> dict:
    dump = booster.dump_model()
    n_trees = n_leaves = n_splits = 0

    def walk(node):
        nonlocal n_leaves, n_splits
        if "split_index" not in node:  # leaf (or a single-leaf tree)
            n_leaves += 1
            return
        n_splits += 1
        walk(node["left_child"])
        walk(node["right_child"])

    for t in dump["tree_info"]:
        n_trees += 1
        walk(t["tree_structure"])
    text = booster.model_to_string().encode("utf-8")
    return {
        "n_trees": n_trees,
        "n_split_nodes": n_splits,
        "n_leaves": n_leaves,
        "n_stored_params": 2 * n_splits + n_leaves,
        "model_file_mb": round(len(text) / 1e6, 3),
        "model_file_gzip_mb": round(len(gzip.compress(text, 9)) / 1e6, 3),
    }


def run(dataset: str, feature_set: str) -> dict:
    fuse = C.results_dir(dataset, "04_fusion")
    sel = C.results_dir(dataset, "05_selection")
    bench = C.results_dir(dataset, "06_benchmark", feature_set)
    card = json.loads((C.results_dir(dataset, "07_final", feature_set) / "model_card.json").read_text())

    data = F.load(fuse)
    X, y = data["X"], data["y"]
    split = data["manifest"]["split"].to_numpy()
    cfg = json.loads((bench / "best_config.json").read_text())
    model, strategy = cfg["best_model"], cfg["best_strategy"]
    if feature_set == "8_consensus":
        X = X[:, np.load(sel / "mask_consensus.npy")]

    tr, te = split == "train", split == "test"
    print(f"{dataset}/{feature_set}: {model} + {strategy} on {X.shape[1]} features, "
          f"{tr.sum():,} train rows", flush=True)
    res = EV.fit_predict(model, strategy, X[tr], y[tr], X[te], y[te], C.SEED)
    m = res["metrics"]
    booster = res["pipeline"].named_steps["clf"].booster_
    row = {
        "dataset": dataset, "feature_set": feature_set, "model": model,
        "balancing": strategy, "n_features": int(X.shape[1]),
        **booster_stats(booster),
        "accuracy": round(100 * m["accuracy"], 3),
        "f1_macro": round(100 * m["f1_macro"], 3),
        "card_accuracy": card["test"]["accuracy"],
        "card_f1_macro": card["test"]["f1_macro"],
    }
    print(json.dumps(row, indent=1), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["cao", "tomasov"])
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "shared" / "17_model_footprint"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        for fs in ("8_consensus", "full_1002"):
            rows.append(run(ds, fs))
            pd.DataFrame(rows).to_csv(out / "footprint.csv", index=False)
    (out / "footprint.json").write_text(json.dumps(rows, indent=1))
    print("written to", out)


if __name__ == "__main__":
    main()
