"""Compare two Results trees file by file - e.g. the Jetson's against a re-run.

    python _compare_results.py ..\\Results ..\\Rerun\\Results              # print summary
    python _compare_results.py ..\\Results ..\\Rerun\\Results --md OUT.md  # also write a report
    python _compare_results.py A B --only cao/01_splits shared/09        # restrict to sub-paths

Every table (.csv), manifest (.parquet), record (.json), text file and array
(.npy / .npz, up to 2 GB) present in both trees is compared value by value:

    identical  same values
    close      numeric values agree within rtol 1e-3 (plus atol 1e-6)
    different  anything else - the largest discrepancies are listed

Machine-specific values are excluded from the verdict: file paths, and timing
columns/keys (seconds, ms, throughput, wall clock), which are reported apart.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-3, 1e-6
MAX_ARRAY_BYTES = 2 * 1024**3
TIMING = re.compile(r"(^|[_\s(])(sec|secs|seconds|s|ms|time|times|timing|latency|throughput|"
                    r"wall|elapsed|duration|clock|per_second|win_per_s)([_\s)]|$)", re.I)
MACHINE = re.compile(r"(^|_)(path|paths|python|platform|executable|host|n_jobs|cpu|device|"
                     r"finished|date|timestamp)(_|$)", re.I)
PATHLIKE = re.compile(r"^[A-Za-z]:[\\/]|^/mnt/|^/home/|\\Users\\")


@dataclass
class Verdict:
    rel: str
    status: str                 # identical | close | different | skipped
    detail: str = ""
    timing: list[str] = field(default_factory=list)


def _is_excluded(name: str) -> bool:
    return bool(MACHINE.search(name))


def _num_close(a: np.ndarray, b: np.ndarray) -> tuple[bool, bool, float, float]:
    """(all equal, all close, max |diff|, max rel diff) over finite pairs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    equal = bool(np.array_equal(a, b, equal_nan=True))
    close = bool(np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True).all())
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(a - b)
        r = d / np.maximum(np.abs(b), ATOL)
    d = d[np.isfinite(d)]
    r = r[np.isfinite(r)]
    return equal, close, float(d.max()) if d.size else 0.0, float(r.max()) if r.size else 0.0


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> tuple[str, str, list[str]]:
    if a.shape != b.shape:
        return "different", f"shape {a.shape} vs {b.shape}", []
    if list(a.columns) != list(b.columns):
        missing = sorted(set(map(str, a.columns)) ^ set(map(str, b.columns)))
        return "different", f"columns differ: {missing[:6]}", []
    status, notes, timing = "identical", [], []
    for col in a.columns:
        name = str(col)
        if _is_excluded(name):
            continue
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            eq, close, dmax, rmax = _num_close(x.to_numpy(), y.to_numpy())
            if TIMING.search(name):
                if not eq:
                    timing.append(f"{name} (max rel {rmax:.2g})")
                continue
            if eq:
                continue
            n_bad = int((~np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=RTOL, atol=ATOL,
                                     equal_nan=True)).sum())
            if close:
                status = "close" if status == "identical" else status
                notes.append(f"{name}: max |d| {dmax:.3g}")
            else:
                status = "different"
                notes.append(f"{name}: {n_bad}/{len(x)} rows off, max |d| {dmax:.3g}")
        else:
            xs, ys = x.astype(str), y.astype(str)
            if xs.str.contains(PATHLIKE).any() or ys.str.contains(PATHLIKE).any():
                continue
            n_bad = int((xs.to_numpy() != ys.to_numpy()).sum())
            if n_bad:
                status = "different"
                first = int(np.flatnonzero(xs.to_numpy() != ys.to_numpy())[0])
                notes.append(f"{name}: {n_bad}/{len(x)} differ (e.g. {xs.iloc[first]!r} vs {ys.iloc[first]!r})")
    order = {"different": 0, "close": 1}
    notes.sort(key=lambda s: order.get("different" if "rows off" in s or "differ" in s else "close", 2))
    return status, "; ".join(notes[:4]), timing


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_json(a, b) -> tuple[str, str, list[str]]:
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    if set(fa) != set(fb):
        diff = sorted(set(fa) ^ set(fb))
        return "different", f"keys differ: {diff[:4]}", []
    status, notes, timing = "identical", [], []
    for k in fa:
        x, y = fa[k], fb[k]
        leaf = re.sub(r"\[\d+\]", "", k).split(".")[-1]
        if _is_excluded(leaf) or (isinstance(x, str) and PATHLIKE.search(x)):
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            eq, close, dmax, _ = _num_close(np.array([x]), np.array([y]))
            if TIMING.search(leaf):
                if not eq:
                    timing.append(leaf)
                continue
            if not eq:
                if close:
                    status = "close" if status == "identical" else status
                else:
                    status = "different"
                    notes.append(f"{k}: {x!r} vs {y!r}")
        elif x != y:
            status = "different"
            notes.append(f"{k}: {str(x)[:40]!r} vs {str(y)[:40]!r}")
    return status, "; ".join(notes[:4]), timing


def compare_arrays(a: np.ndarray, b: np.ndarray) -> tuple[str, str]:
    if a.shape != b.shape:
        return "different", f"shape {a.shape} vs {b.shape}"
    if a.dtype == object or b.dtype == object or a.dtype.kind in "USb" or b.dtype.kind in "USb":
        n_bad = int((np.asarray(a).astype(str) != np.asarray(b).astype(str)).sum())
        return ("identical", "") if not n_bad else ("different", f"{n_bad}/{a.size} entries differ")
    # Chunk along the first axis so 700 MB feature matrices stay cheap.
    step = max(1, int(5e7 // max(1, a[0:1].size)))
    n_bad = 0
    dmax = 0.0
    equal = True
    for i in range(0, a.shape[0] if a.ndim else 1, step):
        x = np.asarray(a[i:i + step] if a.ndim else a, dtype=np.float64)
        y = np.asarray(b[i:i + step] if b.ndim else b, dtype=np.float64)
        equal &= bool(np.array_equal(x, y, equal_nan=True))
        bad = ~np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True)
        n_bad += int(bad.sum())
        d = np.abs(x - y)
        d = d[np.isfinite(d)]
        if d.size:
            dmax = max(dmax, float(d.max()))
    if equal:
        return "identical", ""
    if not n_bad:
        return "close", f"max |d| {dmax:.3g}"
    return "different", f"{n_bad}/{a.size} values off ({100 * n_bad / a.size:.3f}%), max |d| {dmax:.3g}"


def compare_file(pa: Path, pb: Path, rel: str) -> Verdict:
    ext = pa.suffix.lower()
    try:
        if ext == ".csv":
            s, d, t = compare_frames(pd.read_csv(pa), pd.read_csv(pb))
        elif ext == ".parquet":
            s, d, t = compare_frames(pd.read_parquet(pa), pd.read_parquet(pb))
        elif ext == ".json":
            s, d, t = compare_json(json.loads(pa.read_text(encoding="utf-8")),
                                   json.loads(pb.read_text(encoding="utf-8")))
        elif ext == ".npy":
            if pa.stat().st_size > MAX_ARRAY_BYTES:
                return Verdict(rel, "skipped", f"{pa.stat().st_size / 1e9:.1f} GB cache")
            a = np.load(pa, mmap_mode="r", allow_pickle=True)
            b = np.load(pb, mmap_mode="r", allow_pickle=True)
            s, d = compare_arrays(a, b)
            t = []
        elif ext == ".npz":
            with np.load(pa, allow_pickle=True) as za, np.load(pb, allow_pickle=True) as zb:
                if sorted(za.files) != sorted(zb.files):
                    return Verdict(rel, "different", "array names differ")
                res = {k: compare_arrays(za[k], zb[k]) for k in za.files}
            bad = {k: v for k, v in res.items() if v[0] != "identical"}
            s = "identical" if not bad else ("different" if any(v[0] == "different" for v in bad.values()) else "close")
            d = "; ".join(f"{k}: {v[1]}" for k, v in list(bad.items())[:4])
            t = []
        elif ext == ".txt":
            same = pa.read_text(encoding="utf-8", errors="replace") == pb.read_text(encoding="utf-8", errors="replace")
            s, d, t = ("identical" if same else "different"), "", []
        else:
            return Verdict(rel, "skipped", ext)
    except Exception as exc:                      # unreadable / malformed - report, keep going
        return Verdict(rel, "different", f"could not compare: {exc!r}"[:160])
    return Verdict(rel, s, d, t)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    a_root, b_root = Path(argv[0]), Path(argv[1])
    md = Path(argv[argv.index("--md") + 1]) if "--md" in argv else None
    only = argv[argv.index("--only") + 1:] if "--only" in argv else []
    exts = {".csv", ".parquet", ".json", ".npy", ".npz", ".txt"}

    rels_a = {p.relative_to(a_root).as_posix() for p in a_root.rglob("*") if p.is_file() and p.suffix.lower() in exts}
    rels_b = {p.relative_to(b_root).as_posix() for p in b_root.rglob("*") if p.is_file() and p.suffix.lower() in exts}
    keep = (lambda r: any(r.startswith(o) for o in only)) if only else (lambda r: "smoke" not in r)
    common = sorted(r for r in rels_a & rels_b if keep(r))
    only_a = sorted(r for r in rels_a - rels_b if keep(r))
    only_b = sorted(r for r in rels_b - rels_a if keep(r))

    verdicts = [compare_file(a_root / r, b_root / r, r) for r in common]
    counts = pd.Series([v.status for v in verdicts]).value_counts().to_dict() if verdicts else {}
    lines = [f"# Results comparison\n", f"- A (reference): `{a_root}`", f"- B (re-run): `{b_root}`",
             f"- compared {len(common)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
             f"- only in A: {len(only_a)} | only in B: {len(only_b)}\n",
             "| file | verdict | detail |", "|---|---|---|"]
    for v in verdicts:
        detail = v.detail + (f" (timing differs: {', '.join(v.timing[:3])})" if v.timing else "")
        lines.append(f"| `{v.rel}` | {v.status} | {detail.replace('|', '/')} |")
    if only_a:
        lines += ["", "## Only in A", *[f"- `{r}`" for r in only_a]]
    if only_b:
        lines += ["", "## Only in B", *[f"- `{r}`" for r in only_b]]
    text = "\n".join(lines)
    print(text)
    if md:
        md.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
