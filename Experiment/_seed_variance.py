"""Seed variance of notebook 16's neural baselines on Tomasov's leakage-safe split.

Notebook 16 trains each recent architecture once, at seed C.SEED, and two of its
conclusions rest on those single runs: the openclose verdict (R2C2), which turns
on whether any model reaches openclose F1 30 on our partition, and the margin of
our pipeline over the strongest neural model.  Both move between machines at the
same seed - S2-DyGNN's openclose F1 is 62.5 on the Jetson and 0.0 on the laptop -
so this retrains the models that decide them under further seeds.  Nothing else
changes: the same manifest, the representations notebook 16 cached, and the
notebook's own training code and budget, read from the notebook itself.

Each finished run writes run_<model>_seed<N>.json to OUT_DIR, with its per-class
report, confusion matrix and test predictions alongside, and a re-run skips the
runs already there.  After every run, seed_runs.csv, seed_summary.csv and
seed_verdict.json are rebuilt from all runs so far, notebook 16's own included.
The log is OUT_DIR/live.log, or live_<model>.log when one model is named, so two
processes - one per model - can run side by side.

    $env:DAS_RESULTS_DIR = "..\\Rerun\\fixed\\Results"     # PowerShell
    python _seed_variance.py                    # S2-DyGNN then DAStatFormer, seeds C.SEED+1..+3
    python _seed_variance.py DAStatFormer       # one model, e.g. beside a second process
    python _seed_variance.py --smoke            # every code path on a tiny subset; writes nothing
    python _seed_variance.py --summary          # rebuild the summary from the runs so far

Follow a log with Get-Content <log> -Wait -Tail 40 (PowerShell) or tail -f.
"""
from __future__ import annotations

import ast
import datetime as dt
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _run_all import keep_awake
from dasfe import config as C, evaluate as EV, preprocess as pp, sota as SO, splits as SPL

NOTEBOOK = HERE / "16_sota_head_to_head.ipynb"
DS, ARM = "tomasov", "A_blocked"
MODELS = ("S2-DyGNN", "DAStatFormer")     # the single runs the verdict and the margin hang on
BASE_SEED = C.SEED
SEEDS = tuple(BASE_SEED + k for k in (1, 2, 3))
# Taken from notebook 16 as it stands: how it builds and audits the manifest, its
# training harness, and the settings that fix the budget and the topology.
FROM_NOTEBOOK = {"stratified_random_split", "audit_split", "build_manifest",
                 "ArrayRows", "fit_scaler", "predict", "train_sota",
                 "PER_CLASS", "NODES", "EPOCHS", "PATIENCE", "PREPROCESS"}
CACHE_STEM = {"patch": "patches", "fused1d": "dasvit", "stats": "stats"}   # notebook 16's cache names
OPENCLOSE_LIMIT, CONFOUND_GAP = 30.0, 25.0      # notebook 16's verdict thresholds (section 6)


class Live:
    """Console plus a log file, flushed on every write, so a long run can be followed."""

    def __init__(self, path: Path | None):
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def write(self, text: str) -> int:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        if self.fh:
            self.fh.write(text)
            self.fh.flush()
        return len(text)

    def flush(self) -> None:
        sys.__stdout__.flush()


def tag(model: str) -> str:
    return model.replace(" ", "_").replace("/", "-")      # notebook 16's file-name form


def notebook_code() -> str:
    """The definitions and settings in FROM_NOTEBOOK, exactly as notebook 16 has them."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    keep, found = [], set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)):
                name = node.targets[0].id
            else:
                continue
            if name in FROM_NOTEBOOK:
                keep.append(node)
                found.add(name)
    if FROM_NOTEBOOK - found:
        raise RuntimeError(f"{NOTEBOOK.name} no longer defines {sorted(FROM_NOTEBOOK - found)}")
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def harness() -> dict:
    """Notebook 16's code, run against the modules its setup cell imports."""
    ns = {"json": json, "time": time, "gc": gc, "shutil": shutil, "Path": Path,
          "np": np, "pd": pd, "torch": torch, "nn": nn,
          "DataLoader": DataLoader, "Dataset": Dataset,
          "C": C, "EV": EV, "pp": pp, "SPL": SPL, "SO": SO,
          "dev": "cuda" if torch.cuda.is_available() else "cpu",
          "display": lambda df: print(df.to_string(index=False))}
    exec(compile(notebook_code(), NOTEBOOK.name, "exec"), ns)
    return ns


def save(obj, path: Path, index: bool = False) -> None:
    """Write via a temporary file, so a reader never sees half a file.  One held open
    elsewhere (Excel locks it) is skipped with a warning rather than ending the run."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(tmp, index=index)
        else:
            tmp.write_text(json.dumps(obj, indent=2, default=lambda v: v.item()), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"  could not write {path.name}: {exc}")
        tmp.unlink(missing_ok=True)


def openclose_f1(pc: pd.DataFrame) -> float:
    hit = pc[pc["class"] == "openclose"]                    # as notebook 16 reads it
    return round(100 * float(hit["f1-score"].iloc[0]), 3) if len(hit) else float("nan")


def reference_runs(ref_dir: Path) -> pd.DataFrame:
    """Notebook 16's own runs on this dataset and arm, at seed C.SEED."""
    prog = pd.read_csv(ref_dir / "head_to_head_progress.csv")
    ref = prog[(prog.dataset == DS) & (prog.arm == ARM)].copy()
    ref["seed"] = BASE_SEED
    ref["source"] = "notebook 16"
    ref["openclose_f1"] = [openclose_f1(pd.read_csv(ref_dir / f"per_class_{DS}_{ARM}_{tag(m)}.csv"))
                           for m in ref.model]
    return ref.reset_index(drop=True)


def summarise(out: Path | None, ref_dir: Path, runs: list[dict] | None = None) -> None:
    """Rebuild seed_runs.csv, seed_summary.csv and seed_verdict.json from every run so far."""
    if runs is None:
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        print("no seed runs yet")
        return
    new = pd.DataFrame(runs)
    ref = reference_runs(ref_dir)
    seeded = [m for m in ref.model if m in set(new.model)]          # notebook 16's order
    order = {m: i for i, m in enumerate(seeded)}
    allr = pd.concat([ref[ref.model.isin(seeded)], new], ignore_index=True)
    allr = allr.sort_values(["model", "seed"], key=lambda s: s.map(order) if s.name == "model" else s)
    first = ["model", "seed", "source", "accuracy", "balanced_accuracy", "f1_macro", "openclose_f1",
             "val_f1_macro", "epochs_run", "best_epoch", "train_seconds"]
    allr = allr[first + [c for c in allr.columns if c not in first]]

    g = allr.groupby("model", sort=False)
    summ = g[["accuracy", "f1_macro", "openclose_f1"]].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.insert(0, "seeds", g["seed"].agg(lambda s: " ".join(str(int(v)) for v in s)))
    summ.insert(0, "n_runs", g.size())
    summ["epochs_run_mean"] = g["epochs_run"].mean()
    summ = summ.round(3).reset_index()

    # Notebook 16's verdict, re-taken with each seed: the re-seeded models take that
    # seed's run, while our pipeline and the models run once keep the row it measured.
    bd = pd.read_csv(ref_dir / "openclose_breakdown.csv")
    once = bd[(bd.arm == ARM) & ~bd.model.isin(seeded)]
    best_b = float(bd.loc[bd.arm == "B_random", "openclose_f1"].max())
    ours = bd[bd.model.str.startswith("Ours")].iloc[0]

    def verdict(best_a: float) -> str:
        if best_a >= OPENCLOSE_LIMIT:
            return "REPRESENTATION LIMIT"
        return "DATASET CONFOUND" if best_b - best_a >= CONFOUND_GAP else "MIXED"

    def spread(x: pd.Series) -> dict:
        return {"mean": round(x.mean(), 3), "sd": round(x.std(), 3) if len(x) > 1 else None,
                "min": round(x.min(), 3), "max": round(x.max(), 3)}

    per_model = {}
    for m, s in g:
        d_f1, d_acc = ours.f1_macro - s.f1_macro, ours.accuracy - s.accuracy
        per_model[m] = {"n_runs": len(s),
                        "openclose_f1": spread(s.openclose_f1),
                        "runs_reaching_openclose_limit": f"{int((s.openclose_f1 >= OPENCLOSE_LIMIT).sum())}/{len(s)}",
                        "ours_minus_model_f1_macro": spread(d_f1),
                        "ours_minus_model_accuracy": spread(d_acc),
                        "ours_ahead_on_f1_macro": f"{int((d_f1 > 0).sum())}/{len(s)}"}
    per_seed = []
    for seed, s in allr.groupby("seed"):
        if set(s.model) != set(seeded):
            continue                                   # until every re-seeded model has this seed
        cand = pd.concat([once, s], ignore_index=True)
        top = cand.loc[cand.openclose_f1.idxmax()]
        per_seed.append({"seed": int(seed), "best_model": top.model,
                         "best_openclose_f1": round(float(top.openclose_f1), 3),
                         "verdict": verdict(float(top.openclose_f1))})
    means = pd.concat([once.set_index("model").openclose_f1, g.openclose_f1.mean()])
    report = {
        "dataset": DS, "arm": ARM,
        "ours": {"model": ours.model, "accuracy": float(ours.accuracy),
                 "f1_macro": float(ours.f1_macro), "openclose_f1": float(ours.openclose_f1)},
        "models": per_model,
        "openclose_verdict": {
            "rule": (f"best arm-A openclose F1 >= {OPENCLOSE_LIMIT:g}: REPRESENTATION LIMIT; otherwise best "
                     f"arm-B minus best arm-A >= {CONFOUND_GAP:g}: DATASET CONFOUND; otherwise MIXED"),
            "best_f1_random_protocol": best_b,
            "held_at_notebook_16": {r.model: r.openclose_f1 for r in once.itertuples()},
            "per_seed": per_seed,
            "on_seed_means": {"best_model": means.idxmax(),
                              "best_openclose_f1": round(float(means.max()), 3),
                              "verdict": verdict(float(means.max()))}},
    }

    print("\n--- all runs so far")
    print(summ.to_string(index=False))
    for v in per_seed:
        print(f"  seed {v['seed']}: best openclose F1 {v['best_openclose_f1']:.2f} ({v['best_model']}) "
              f"-> {v['verdict']}")
    om = report["openclose_verdict"]["on_seed_means"]
    print(f"  on seed means: best openclose F1 {om['best_openclose_f1']:.2f} ({om['best_model']}) "
          f"-> {om['verdict']}")
    for m, d in per_model.items():
        print(f"  ours minus {m}: macro-F1 {d['ours_minus_model_f1_macro']['mean']:+.2f} on average, "
              f"ours ahead in {d['ours_ahead_on_f1_macro']} runs")
    if out:
        save(allr, out / "seed_runs.csv")
        save(summ, out / "seed_summary.csv")
        save(report, out / "seed_verdict.json")


def main(argv: list[str]) -> int:
    smoke, only_summary = "--smoke" in argv, "--summary" in argv
    models = [a for a in argv if not a.startswith("--")] or list(MODELS)
    unknown = [m for m in models if m not in SO.SOTA_MODELS]
    if unknown:
        print(__doc__)
        print(f"unknown model(s) {unknown}; notebook 16 has {list(SO.SOTA_MODELS)}")
        return 2
    ref_dir = C.results_dir("shared", "16_sota_head_to_head")
    out = None if smoke else C.results_dir("shared", "16_seed_variance")
    if only_summary:
        summarise(out, ref_dir)
        return 0

    sys.__stdout__.reconfigure(errors="replace")
    log = None if smoke else out / ("live.log" if len(models) > 1 else f"live_{tag(models[0])}.log")
    sys.stdout = sys.stderr = Live(log)
    seeds = SEEDS[:1] if smoke else SEEDS
    ns = harness()
    dev = ns["dev"]
    print(f"\n=== seed runs started {dt.datetime.now():%Y-%m-%d %H:%M} | pid {os.getpid()} "
          f"| results -> {C.RESULTS_DIR}")
    print(f"    {DS} / {ARM} | {', '.join(models)} | seeds {', '.join(map(str, seeds))} "
          f"(notebook 16 ran seed {BASE_SEED}) | {dev} "
          f"{torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    if smoke:
        print("SMOKE MODE: tiny subset, 1 epoch, nothing written. Nothing here is a result.")
    keep_awake()

    sub = ns["build_manifest"](DS)
    sp = sub["split_A"].to_numpy()
    ref = reference_runs(ref_dir)
    if not smoke and ((sp == "train").sum(), (sp == "test").sum()) != (ref.n_train.iloc[0], ref.n_test.iloc[0]):
        raise SystemExit("the arm-A split differs from the one notebook 16 trained on")
    if smoke:
        # A few rows per class and split, through the real caches and the real code.
        keep = sub.groupby(["label", "split_A"]).cumcount() < 24
        sub["split_A"] = np.where(keep, sub["split_A"], "unused")
        ns["EPOCHS"], ns["PATIENCE"] = 1, 1

    cache = {"manifest": sub, "y": sub["label"].to_numpy(), "classes": sorted(sub["label"].unique())}
    for key in sorted({SO.SOTA_MODELS[m]["input"] for m in models}):
        p = ref_dir / f"{CACHE_STEM[key]}_{DS}_{'pp' if ns['PREPROCESS'] else 'raw'}.npy"
        if not p.exists():
            raise SystemExit(f"{p} not found - notebook 16 builds this cache; run it first")
        arr = np.load(p, mmap_mode="r")        # read-only: the cache is shared, never written here
        if len(arr) != len(sub):
            raise SystemExit(f"{p.name} has {len(arr):,} rows, the manifest {len(sub):,}")
        cache[key] = arr
    ns["CACHE"] = {DS: cache}

    failed, todo, smoke_runs = 0, [(m, s) for m in models for s in seeds], []
    for k, (model, seed) in enumerate(todo, 1):
        stem = f"{tag(model)}_seed{seed}"
        if out and (out / f"run_{stem}.json").exists():
            print(f"skip (done): {model} / seed {seed}")
            continue
        r = ref[ref.model == model]
        before = (f"{r.train_seconds.iloc[0] / 3600:.1f} h, {int(r.epochs_run.iloc[0])} epochs"
                  if len(r) else "not run")
        print(f"\n=== {model} | seed {seed} | run {k}/{len(todo)} | {dt.datetime.now():%Y-%m-%d %H:%M} "
              f"| notebook 16's seed-{BASE_SEED} run took {before}")
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        C.SEED = seed               # train_sota seeds torch with C.SEED; the split above used the default
        try:
            row, yt, yp = ns["train_sota"](model, DS, ARM)
        except Exception as exc:    # as notebook 16 does: report it and go on to the next run
            print(f"  FAILED {type(exc).__name__}: {exc}")
            failed += 1
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            continue
        finally:
            C.SEED = BASE_SEED
        classes = cache["classes"]
        pc = EV.per_class_report(yt, yp, labels=classes)
        row.update(seed=seed, source="seed run", openclose_f1=openclose_f1(pc),
                   finished=dt.datetime.now().isoformat(timespec="seconds"))
        if dev == "cuda":
            row["peak_gpu_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
        print(f"  -> openclose F1 {row['openclose_f1']:.2f}"
              + (f" | peak GPU memory {row['peak_gpu_mb']:,} MB" if dev == "cuda" else ""))
        if out:
            i_te = np.flatnonzero(sub["split_A"].to_numpy() == "test")
            save(pc, out / f"per_class_{stem}.csv")
            save(EV.confusion(yt, yp, labels=classes, normalize="true"), out / f"confusion_{stem}.csv",
                 index=True)
            save(pd.DataFrame({"sample_id": sub["sample_id"].to_numpy()[i_te], "y_true": yt, "y_pred": yp}),
                 out / f"predictions_{stem}.csv")
            save(row, out / f"run_{stem}.json")      # last: it marks the run complete
        else:
            smoke_runs.append(row)
        summarise(out, ref_dir, None if out else smoke_runs)
    print(f"\n=== seed runs finished {dt.datetime.now():%Y-%m-%d %H:%M}: "
          f"{len(todo) - failed} done, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
