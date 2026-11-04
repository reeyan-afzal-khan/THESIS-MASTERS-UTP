"""Write notebook 12's corrected spatial-confound verdict (verdict_corrected.json).

The executed notebook 12 writes `verdict.json` with its original rule.  The
corrected rule - drop degenerate recording probes, and score spatial against
the strongest other domain rather than against chance - lives in the verdict
cell of `notebook_specs/nb_12_spatial_confound_ablation.py`; its output was saved
on the Jetson as `verdict_corrected.json`, the verdict the thesis discusses.
This script applies that cell to the tables notebook 12 already saved, so the
corrected verdict is reproducible from any results tree without re-running the
probes.

    python _corrected_spatial_verdict.py ..\\Rerun\\Results            # writes verdict_corrected.json there
    python _corrected_spatial_verdict.py ..\\Results OUT.json         # regenerate elsewhere, e.g. to check
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = HERE / "notebook_specs" / "nb_12_spatial_confound_ablation.py"
VERDICT_CELL = 8                       # index among the spec's code cells


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    d = Path(argv[0]) / "shared" / "12_spatial_ablation"
    spec = importlib.util.spec_from_file_location("nb12", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cell = [src for kind, src in mod.CELLS if kind == "code"][VERDICT_CELL]
    loro = pd.read_csv(d / "loro_with_without_spatial.csv")
    piv = loro.pivot(index="held_out", columns="arm", values="accuracy")   # as notebook 12 builds it
    piv["delta_pp"] = (piv["without spatial (941)"] - piv["with spatial (1002)"]).round(2)
    with tempfile.TemporaryDirectory() as tmp:          # the cell also writes verdict.json
        ns = {
            "np": np, "pd": pd, "json": json, "display": print,
            "DATASETS": ("tomasov", "cao"),
            "lodo": pd.read_csv(d / "leave_one_domain_out.csv"),
            "loro": loro, "piv": piv,
            "probe": pd.read_csv(d / "recording_identity_probe.csv"),
            "out_dir": Path(tmp),
        }
        exec(compile(cell, f"{SPEC.name}[code cell {VERDICT_CELL}]", "exec"), ns)
    out = Path(argv[1]) if len(argv) > 1 else d / "verdict_corrected.json"
    out.write_text(json.dumps(ns["verdict"], indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
