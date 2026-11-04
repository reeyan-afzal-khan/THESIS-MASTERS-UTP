"""Reconcile the two Tomasov test-window counts and re-verify zero overlap (R1C9).

Reviewer R1C9 asks why Section 4.1 reports 31,028 test windows in the leakage
assessment while Table 6 reports 11,334 in the final partition, and asks that the
zero-overlap verification be repeated on the exact final test set.

This script answers both from stored artefacts. It fits no models and takes
seconds. Paths are resolved relative to the script so it runs on either machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = Path(os.environ.get("DAS_RESULTS_DIR", ROOT / "Results"))   # same override as dasfe.config
sys.path.insert(0, str(HERE))

WIN_LEN, WIN_HOP = 8192, 2048
BLOCKS_PER_WINDOW = WIN_LEN // WIN_HOP           # 4
GUARD_BLOCKS = BLOCKS_PER_WINDOW                 # one full window length


def contaminated(df: pd.DataFrame, rec="recording_id", blk="block", split="split") -> tuple[int, int]:
    """Count test windows whose raw-sample span intersects the training footprint."""
    hit = tot = 0
    for _, sub in df.groupby(rec, sort=False):
        n = int(sub[blk].max()) + BLOCKS_PER_WINDOW + 1
        cov = np.zeros(n, dtype=bool)
        b_tr = sub.loc[sub[split] == "train", blk].to_numpy()
        for off in range(BLOCKS_PER_WINDOW):
            cov[b_tr + off] = True
        b_te = sub.loc[sub[split] == "test", blk].to_numpy()
        if b_te.size == 0:
            continue
        h = np.zeros(b_te.size, dtype=bool)
        for off in range(BLOCKS_PER_WINDOW):
            h |= cov[b_te + off]
        hit += int(h.sum())
        tot += int(b_te.size)
    return hit, tot


def pairwise_disjoint(df: pd.DataFrame, rec="recording_id", blk="block", split="split") -> list:
    """Blocks read by each split must be pairwise disjoint inside every recording."""
    bad = []
    for name, sub in df.groupby(rec, sort=False):
        n = int(sub[blk].max()) + BLOCKS_PER_WINDOW + 1
        cov = {}
        for s, s2 in sub.groupby(split, sort=False):
            c = np.zeros(n, dtype=bool)
            b = s2[blk].to_numpy()
            for off in range(BLOCKS_PER_WINDOW):
                c[b + off] = True
            cov[s] = c
        ks = list(cov)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                shared = int((cov[ks[i]] & cov[ks[j]]).sum())
                if shared:
                    bad.append((name, ks[i], ks[j], shared))
    return bad


def main() -> int:
    final_p = RESULTS / "tomasov" / "01_splits" / "split_manifest.parquet"
    if not final_p.exists():
        print("not found:", final_p)
        return 1
    final = pd.read_parquet(final_p)

    print("=" * 72)
    print("R1C9  Reconciliation of Tomasov test-window counts")
    print("=" * 72)

    print("\n-- Final partition actually used for all reported results --")
    counts = final["split"].value_counts()
    for s in ("train", "val", "test"):
        print(f"   {s:<6} {counts.get(s, 0):>8,}")
    print(f"   {'TOTAL':<6} {len(final):>8,}")

    n_test_final = int(counts.get("test", 0))

    # Try to rebuild the pre-decimation guarded population for the reconciliation.
    inv_p = RESULTS / "tomasov" / "00_inventory" / "manifest.parquet"
    pre_test = None
    if inv_p.exists():
        try:
            from dasfe import splits as S
            inv = pd.read_parquet(inv_p)
            assigned = S.blocked_split_by_recording(
                inv, recording_col="recording_id", block_col="block",
                n_segments=10, fractions=(0.8, 0.1, 0.1), seed=42)
            inv = inv.join(assigned)
            keep = S.apply_temporal_guard(inv, split_col="split",
                                          recording_col="recording_id", block_col="block")
            guarded = inv[keep]
            pre_test = int((guarded["split"] == "test").sum())
            print("\n-- Pre-decimation population (guard band applied, no decimation) --")
            gc = guarded["split"].value_counts()
            for s in ("train", "val", "test"):
                print(f"   {s:<6} {gc.get(s, 0):>8,}")
            print(f"   {'TOTAL':<6} {len(guarded):>8,}")
        except Exception as exc:
            print(f"\n   (could not rebuild pre-decimation set: {type(exc).__name__}: {exc})")

    if pre_test:
        print(f"\n-- Reconciliation --")
        print(f"   pre-decimation  test windows : {pre_test:,}")
        print(f"   post-decimation test windows : {n_test_final:,}")
        print(f"   removed by decimation        : {pre_test - n_test_final:,} "
              f"({100*(pre_test-n_test_final)/pre_test:.1f}%)")
        print("   Decimation caps the three most populous classes and is applied AFTER")
        print("   partitioning, so it removes rows from every split including test.")

    print("\n-- Zero-overlap verification on the EXACT final test set --")
    hit, tot = contaminated(final)
    print(f"   test windows examined                    : {tot:,}")
    print(f"   sharing raw samples with a training window: {hit:,} ({100*hit/max(tot,1):.4f}%)")

    bad = pairwise_disjoint(final)
    print(f"   recordings with cross-split block overlap : {len(bad)}")

    groups_multi = int((final.groupby("group_id")["split"].nunique() > 1).sum()) \
        if "group_id" in final.columns else -1
    print(f"   split groups appearing in >1 split         : {groups_multi}")

    ok = (hit == 0) and (not bad) and (groups_multi == 0)
    print("\n" + "=" * 72)
    print("VERDICT:", "PASS - zero overlap confirmed on the final 11,334-window test set"
          if ok else "FAIL - see above")
    print("=" * 72)

    out = RESULTS / "tomasov" / "01_splits" / "R1C9_reconciliation.txt"
    out.write_text(
        f"pre_decimation_test_windows={pre_test}\n"
        f"post_decimation_test_windows={n_test_final}\n"
        f"contaminated_on_final_test={hit}\n"
        f"cross_split_block_overlaps={len(bad)}\n"
        f"groups_in_multiple_splits={groups_multi}\n",
        encoding="utf-8")
    print("\nwrote", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
