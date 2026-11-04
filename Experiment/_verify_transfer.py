"""Confirm the Jetson is running the fixed code, not a stale copy.

Run this on the Jetson after copying and before starting notebooks 14-16.

The trap this exists for: `dasfe/__pycache__` can serve bytecode compiled from an
older `robust.py`, so the notebook imports the unfixed function and the bug
reappears with no error message. That already cost one session.

    python _verify_transfer.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

checks: list[tuple[str, bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "", note: str = "") -> None:
    """detail is shown only on failure; note only on success."""
    checks.append((name, bool(ok), detail, note))


# --------------------------------------------------------------------------
# 1. Stale bytecode
# --------------------------------------------------------------------------
pyc = list((HERE / "dasfe" / "__pycache__").glob("*.pyc"))
check("dasfe/__pycache__ removed", not pyc,
      f"{len(pyc)} .pyc file(s) present - run: rm -rf dasfe/__pycache__" if pyc else "")

# --------------------------------------------------------------------------
# 2. dasfe.robust carries both notebook-14 fixes
# --------------------------------------------------------------------------
try:
    from dasfe import robust as R

    src_boot = inspect.getsource(R.group_bootstrap_ci)
    check("robust: bootstrap scores over present classes",
          "frac_replicates_all_classes" in src_boot and "estimable" in src_boot,
          "group_bootstrap_ci is the old version")

    check("robust: multigroup_classes present",
          hasattr(R, "multigroup_classes"),
          "leave_one_group_out_scores has not been updated")

    src_loo = inspect.getsource(R.leave_one_group_out_scores)
    check("robust: LOO scores macro-F1 over present classes",
          "f1_macro_union" in src_loo and "restrict_to_multigroup" in src_loo,
          "leave_one_group_out_scores is the old version")

    # min_classes=2 silently yields zero Tomasov units, because every recording
    # holds exactly one class.  That is what produced KeyError: 'held_out'.
    check("robust: single-class folds allowed (min_classes=1)",
          inspect.signature(R.leave_one_group_out_scores)
          .parameters["min_classes"].default == 1,
          "min_classes defaults to 2 - Tomasov will produce no units at all")

    src_pair = inspect.getsource(R.paired_group_comparison)
    check("robust: paired test tolerates an empty result",
          "held_out" in src_pair and "return empty" in src_pair,
          "paired_group_comparison will raise KeyError on an empty frame")
except Exception as exc:                                   # noqa: BLE001
    check("dasfe.robust imports", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------
# 3. dasfe.sota has all six architectures
# --------------------------------------------------------------------------
try:
    from dasfe import sota as SO

    want = {"S2-DyGNN", "DASViT1D", "DAStatFormer", "CNN-LSTM", "GCN", "GAT"}
    have = set(SO.SOTA_MODELS)
    check("sota: six architectures registered", want <= have,
          f"missing {sorted(want - have)}" if want - have else "")
    check("sota: comparator reference figures present",
          hasattr(SO, "CMP_PUBLISHED") and hasattr(SO, "SPATIAL_CAPABILITY"))
except Exception as exc:                                   # noqa: BLE001
    check("dasfe.sota imports", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------
# 4. The notebooks themselves
# --------------------------------------------------------------------------
EXPECT = {
    "14_repeated_splits_and_bootstrap.ipynb": [
        ("REUSE_CACHED", "cache reuse switch"),
        ("Clip the error bars at zero", "plot no longer crashes on a negative arm"),
        ("Reconciling the 98.90 to 57.82 drop", "LODO reconciliation section"),
    ],
    "15_pipeline_protocol_attribution.ipynb": [
        ("Select rows once", "row selection hoisted out of the config loop"),
        ("One comparison to read carefully", "consensus dimensionality caveat"),
    ],
    "16_sota_head_to_head.ipynb": [
        ("PREPROCESS = True", "preprocessing matched to the feature pipeline"),
        ("audit_split(sub, ds)", "arm A leakage audit"),
        ("arm A is not leakage-safe", "audit fails loudly"),
        ('"S2-DyGNN", "DASViT1D", "DAStatFormer", "GCN", "GAT"',
         "five models, CNN-LSTM dropped"),
        ("PER_CLASS = None", "full manifest, not a capped sample"),
        ("EPOCHS    = 30", "30 epochs for every architecture"),
        ("PATIENCE  = 10", "early stopping patience 10"),
        ("epoch {ep+1:>3}/{EPOCHS}", "per-epoch progress printed"),
        ("SAVE = not SMOKE", "smoke writes nothing"),
        ("head_to_head_progress.csv", "resumable across runs"),
        ("def preflight", "disk pre-flight check"),
        ("model_card.json", "our figures read, not retrained"),
        ("online_ms_per_window", "true online cost compared"),
    ],
}

for name, wants in EXPECT.items():
    p = HERE / name
    if not p.exists():
        check(f"{name} present", False, "file not copied")
        continue
    nb = json.loads(p.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    n_out = sum(1 for c in nb["cells"]
                if c["cell_type"] == "code" and c.get("outputs"))
    check(f"{name} present", True,
          note=f"{len(nb['cells'])} cells, {n_out} with output")
    for token, label in wants:
        check(f"  {name[:2]}: {label}", token in src, f"missing marker: {token!r}")

    # A module alias reassigned at top level becomes a data object and every
    # later call on it raises AttributeError. `P, V, S = build_cache(...)`
    # clobbered `from dasfe import splits as S` and killed a run mid-extraction.
    import re

    aliases = set()
    for imp in re.findall(r"^from dasfe import (.+)$", src, re.M):
        for part in imp.split(","):
            part = part.split("#")[0].strip()
            if part:
                aliases.add(part.split(" as ")[-1].strip()
                            if " as " in part else part)
    shadowed = sorted(
        a for a in aliases
        if re.search(rf"^(?:\w+\s*,\s*)*{re.escape(a)}\s*(?:,\s*\w+)*\s*=(?!=)",
                     src, re.M)
    )
    check(f"  {name[:2]}: no dasfe alias shadowed at top level", not shadowed,
          f"rebound: {shadowed} - rename the alias or the variable")

# --------------------------------------------------------------------------
# 5. Inputs the notebooks need
# --------------------------------------------------------------------------
try:
    from dasfe import config as C

    for ds in ("tomasov", "cao"):
        for stage, fname in (("01_splits", "split_manifest.parquet"),
                             ("04_fusion", "X_multi.npy")):
            p = C.RESULTS_DIR / ds / stage / fname
            check(f"input: {ds}/{stage}/{fname}", p.exists(),
                  "" if p.exists() else f"not found at {p}")
    lodo = C.RESULTS_DIR / "shared" / "08_crossdataset" / "cao_leave_one_date_out.csv"
    check("input: notebook 08 LODO csv (for nb14 section 3b)", lodo.exists(),
          "" if lodo.exists() else "nb14 section 3b will skip")
    cached = C.RESULTS_DIR / "shared" / "14_statistics" / "repeated_splits_raw.csv"
    check("cache: repeated_splits_raw.csv (saves ~65 min)", cached.exists(),
          "" if cached.exists() else "nb14 will recompute test 1")
except Exception as exc:                                   # noqa: BLE001
    check("dasfe.config imports", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------
print("=" * 74)
print("Transfer verification")
print("=" * 74)
fails = 0
for name, ok, detail, note in checks:
    mark = "ok  " if ok else "FAIL"
    if not ok:
        fails += 1
    extra = note if ok else detail
    print(f"  [{mark}] {name}" + (f"   -- {extra}" if extra else ""))
print("=" * 74)
if fails:
    print(f"{fails} check(s) failed. Fix these before running notebooks 14-16.")
else:
    print("All checks passed. Safe to run:  python _run_notebook.py 14")
raise SystemExit(1 if fails else 0)
