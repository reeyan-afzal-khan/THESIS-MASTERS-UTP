"""Rebuild notebooks 10 and 11's head-to-head tables without retraining the CNNs.

Each notebook trains its CNN baseline, saves the results (`res`) as
`*_cnn_results.csv`, and then tabulates them against the model cards notebook 07
wrote.  The CNNs never read the 1,002 features, so after notebook 07 is re-run
- following a change to the feature pipeline, say - only the pipeline rows of
`cao_head_to_head.csv` and `tomasov_head_to_head.csv` are stale.  This runs each
notebook's own head-to-head cell on the CNN results it saved and the model cards
now in the tree, so the CNN rows stay exactly as trained.

    python _refresh_head_to_head.py ..\\Rerun\\fixed\\Results
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {"cao": "10_cao_cnn_baseline.ipynb", "tomasov": "11_tomasov_cnn_baseline.ipynb"}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    os.environ["DAS_RESULTS_DIR"] = str(Path(argv[0]).resolve())
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    from dasfe import config as C

    pd.set_option("display.width", 200)
    for ds, name in NOTEBOOKS.items():
        nb = json.loads((HERE / name).read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cell = next(s for s in code if f"{ds}_head_to_head.csv" in s)
        out_dir = C.results_dir(ds, "10_cnn_baseline")
        # round_trip: the default parser can miss the last bit, and `res` must come
        # back exactly as the notebook saved it.
        ns = {"C": C, "json": json, "np": np, "pd": pd, "out_dir": out_dir,
              "display": lambda df: print(df.to_string(index=False)),
              "res": pd.read_csv(out_dir / f"{ds}_cnn_results.csv", float_precision="round_trip")}
        for s in code:                  # the notebook's own PROTOCOLS setting, if it has one
            m = re.search(r"^PROTOCOLS\s*=.*$", s, re.M)
            if m:
                exec(m.group(0), ns)
        print(f"=== {name}: before")
        print(pd.read_csv(out_dir / f"{ds}_head_to_head.csv").round(3).to_string(index=False))
        print("--- after")
        exec(cell, ns)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
