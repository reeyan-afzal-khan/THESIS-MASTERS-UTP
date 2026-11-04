"""Notebook 16 - head-to-head against 2024+ DAS methods under our protocol.

Answers R1C4 (limited coverage of recent state of the art), R2C1 (S2-DyGNN and
DASViT1D must be rerun on the proposed leakage-controlled manifests) and R2C2
(the openclose failure may be model inadequacy rather than a dataset limitation).

Three recent architectures are reimplemented in `dasfe/sota.py` from their
papers and run here on exactly our manifests, our preprocessing and our metrics,
under both our leakage-safe split and a random split of the same rows.
"""

CELLS = [
    ("md", """
# 16 - Head-to-Head Against Recent DAS Methods, Under Our Protocol

## The three objections this answers

**R1C4** - coverage of the state of the art is too thin; 2024-onward methods
should be compared under identical partitions, preprocessing and metrics.

**R2C1** - S2-DyGNN reports 94.0% accuracy, 86.6% macro-F1 and 55.7% openclose F1
on the same nine-class Tomasov dataset, against our 90.71 / 81.98 / 4.4. The
partitions differ, which is exactly why a true head-to-head is required.

**R2C2** - a recent spatio-spectral graph model reaches openclose F1 above 0.55.
That substantially weakens our claim that the class is unrecoverable once overlap
leakage is removed. The manuscript must separate *dataset confounding* from
*model inadequacy*.

## Why S2-DyGNN cannot be dismissed on leakage grounds

Their Section 6.1 states the protocol explicitly: 8192-sample windows, 2048-sample
shift, chronological allocation of the first 80% to training with validation and
test taking 10% each, **separated by a five-window safety margin**. At a 2048
shift, five windows is 10{,}240 raw samples, which exceeds the 8192-sample window
length. Their partition is therefore sample-disjoint at the boundaries, like ours.

So their 94.0% is not a leakage artefact, and the honest reading is that the gap
is a genuine modelling difference: they aggregate 17 adjacent channels through a
dynamic graph, we compress spatial structure into 61 aggregate scalars. This
notebook tests that reading directly.

## Design

| Model | Family | Input | Published (their protocol) |
|---|---|---|---|
| **S2-DyGNN** | dynamic graph | 17-channel patch, STFT 1024/256 | 94.0 acc / 86.6 F1 / 55.7 openclose |
| **DASViT1D** | transformer on spectra | MFCC+RDFT+DWT fused to 8192 | 93.5 acc / 91.29 F1 |
| **DAStatFormer** | gated multibranch transformer | 24 statistics per channel | 99.93 acc / 99.96 F1 (Tomasov), 99.48 (Cao) |
| **CNN-LSTM** | spatiotemporal (recurrent) | 17-channel patch, LSTM per node | 86.0 acc / 71.4 F1 / 31.2 openclose |
| **GCN** | static graph | 17-channel patch, fixed adjacency | 85.3 acc / 75.5 F1 / 37.2 openclose |
| **GAT** | attention graph | 17-channel patch, learned attention | 81.2 acc / 67.3 F1 / 28.6 openclose |
| **Ours** | multi-domain features | 1002 features, LightGBM | 90.71 acc / 81.98 F1 / 4.4 openclose |

The last three are the comparators S2-DyGNN benchmarks against in its own Table 3.
Including them matters for a specific reason: their published openclose F1 rises
monotonically with how much inter-node structure each model can express - 31.2 for
a recurrent model that treats channels independently, 28.6 and 37.2 for static
graphs, 55.7 for a graph recomputed per frame. If that ordering survives on our
partition, the class is recoverable and the limitation is in our representation.
If it collapses, the ordering was a property of their partition. Either way the
question is settled by a series rather than by one data point.

Every model is trained twice on **identical rows**:

* **arm A** - our leakage-safe split (blocked segments + guard band on Tomasov,
  session-grouped on Cao)
* **arm B** - stratified random split, the protocol DASViT1D and DAStatFormer used

Arm A is the head-to-head R2C1 asks for. Arm B explains their published numbers.

## Which protocol each arm actually uses

Arm A does not re-derive a partition. It reads `split_manifest.parquet` for each
dataset and inherits the assignment used for every other result in this study:
**blocked contiguous segments with guard bands** on Tomasov, **session-grouped**
on Cao. Capping to `PER_CLASS` samples rows within each (label, split) cell and
never reassigns one, so it can only shrink each split's footprint. Section 1 runs
the notebook-02 audits on the exact capped rows and refuses to continue if any
fail, and it measures arm B's contamination rather than assuming it.

## Four asymmetries to declare

1. **Cao arm A is session-grouped, not date-grouped.** That is the protocol this
   paper proposes and the one R2C1 asks the comparison to use, but Section 4.8
   shows that holding out whole acquisition *dates* is a materially harder test.
   The head-to-head therefore measures within-campaign performance for Cao, and
   should be reported as such.
2. **Spatial context differs by design.** S2-DyGNN, CNN-LSTM, GCN and GAT see the
   seventeen adjacent channels their source specifies; on Cao all twelve released
   channels are used. Our own spatial descriptors were computed from a
   thirty-two-channel patch on Tomasov, so on that dataset our method has the
   wider spatial view, not a narrower one.
3. **Preprocessing is matched, which is not the same as faithful.** With
   `PREPROCESS = True` every architecture receives the chain our own features
   receive: demean, 20-2000 Hz band-pass, per-channel z-score. That is what R2C1
   asks for, and it removes an asymmetry the manuscript currently carries - the
   Cao and Tomasov CNN baselines in Sections 4.9 were trained on raw records
   while the 1002 features were not. It does cost the competitors something
   specific: per-channel z-scoring fixes the standard deviation at one, so
   DAStatFormer's amplitude attributes carry no information. Our own time-domain
   features lose exactly the same quantity, so the constraint is symmetric, but
   it should be stated rather than discovered. Set `PREPROCESS = False` for the
   faithful-to-source sensitivity arm.
4. **The matched sample is small.** At 1,500 training windows per class the
   Tomasov head-to-head trains on about 13,500 rows against the 148,229 behind
   the 90.71% headline. Every method here is equally affected, so the comparison
   is fair, but the "Ours" row in this notebook is not the paper's headline number
   and must not be quoted as one.

## Note on faithfulness

Every architectural constant is taken from the source papers and recorded in
`dasfe.sota.ASSUMPTIONS`, including each value we had to supply ourselves. Nothing
is tuned on our data. Where our reconstruction may understate a method, the
notebook says so rather than claiming a win.
"""),
    ("code", """
import sys, json, time, gc
from pathlib import Path
# No machine-specific path: the kernel starts in Experiment/ (or the repo root).
EXPERIMENT_DIR = Path.cwd() if (Path.cwd() / "dasfe").is_dir() else Path.cwd() / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dasfe import config as C, fusion as F, evaluate as EV, preprocess as pp
from dasfe import splits as S
from dasfe import sota as SO

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
DATASETS  = ("tomasov", "cao")
MODELS    = ("S2-DyGNN", "DASViT1D", "DAStatFormer", "CNN-LSTM", "GCN", "GAT")
ARMS      = ("A_blocked", "B_random")

# Windows per class per split.  Every model sees exactly the same rows, so the
# head-to-head is fair at any cap; the cap only bounds runtime and disk.
PER_CLASS = {"train": 1500, "val": 300, "test": 300}

# Spatial neighbourhood.  17 is S2-DyGNN's stated topology; Cao releases 12
# channels per record, so all 12 are used there.
NODES     = {"tomasov": 17, "cao": 12}

# Apply the same preprocessing our own features receive: demean, 20-2000 Hz
# band-pass, per-channel z-score (dasfe.preprocess.preprocess_patch, the chain
# extract.py runs before computing the 1002 features).  R2C1 asks for identical
# preprocessing constraints, so this is True by default.  Set False for a
# sensitivity arm in which each architecture sees the raw signal its own paper
# assumes; the cache is keyed on the setting, so the two do not collide.
PREPROCESS = True

EPOCH_OVERRIDE = None      # e.g. 30 to trim DAStatFormer's 100 epochs
MAX_DISK_GB    = 12.0
REBUILD_CACHE  = False     # True forces re-extraction of the patch cache
# ------------------------------------------------------------------------

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")
out_dir = C.results_dir("shared", "16_sota_head_to_head")
print("output:", out_dir)

PUBLISHED = pd.DataFrame([
    {"model": "S2-DyGNN", "dataset": "tomasov", "accuracy": 94.0,
     "f1_macro": 86.6, "openclose_f1": 55.7,
     "their_protocol": "chronological 80/10/10, 5-window margin (sample-disjoint)"},
    {"model": "DASViT1D", "dataset": "tomasov", "accuracy": 93.5,
     "f1_macro": 91.29, "openclose_f1": np.nan,
     "their_protocol": "80/10/10 over pre-segmented overlapping windows"},
    {"model": "DAStatFormer", "dataset": "tomasov", "accuracy": 99.93,
     "f1_macro": 99.96, "openclose_f1": np.nan,
     "their_protocol": "random 10% validation split"},
    {"model": "DAStatFormer", "dataset": "cao", "accuracy": 99.48,
     "f1_macro": np.nan, "openclose_f1": np.nan,
     "their_protocol": "released 8:2 split (423/441 sessions straddle it)"},
] + [
    # S2-DyGNN's own Table 3, which is the only published source giving
    # openclose F1 for a graded series of spatial architectures.
    {"model": k, "dataset": "tomasov", "accuracy": v["accuracy"],
     "f1_macro": v["f1_macro"], "openclose_f1": v["openclose_f1"],
     "their_protocol": "chronological 80/10/10, 5-window margin (sample-disjoint)"}
    for k, v in SO.CMP_PUBLISHED.items() if k in ("CNN-LSTM", "GCN", "GAT")
])
display(PUBLISHED)
PUBLISHED.to_csv(out_dir / "published_reference.csv", index=False)
_assump = dict(SO.ASSUMPTIONS)
_assump["this_notebook"] = {
    "preprocessing": ("demean, 20-2000 Hz band-pass, per-channel z-score "
                      "(dasfe.preprocess.preprocess_patch) applied to every "
                      "architecture" if PREPROCESS else
                      "raw records, faithful to each source paper"),
    "why": ("R2C1 asks for identical preprocessing constraints. Matching removes "
            "the asymmetry in Sections 4.9, where the CNN baselines saw raw "
            "records and the 1002 features did not."),
    "cost": ("per-channel z-scoring fixes the standard deviation at one, so "
             "amplitude-scale attributes carry no information for any method, "
             "ours included"),
    "arm_A_split": "inherited from split_manifest.parquet, audited in section 1",
    "per_class_cap": PER_CLASS,
    "nodes": NODES,
}
(out_dir / "sota_assumptions.json").write_text(
    json.dumps(_assump, indent=2), encoding="utf-8")
print("wrote the declared-assumptions file for all three reimplementations")
"""),
    ("md", """
## 1. One extraction pass, three representations

Reading the raw data is the expensive step, so it happens once per dataset. The
same pass writes:

* the multi-channel patch, for S2-DyGNN (memory-mapped float16 on disk);
* the fused MFCC+RDFT+DWT vector of the centre channel, for DASViT1D;
* the 24 statistical attributes per channel, for DAStatFormer.

The patch cache is memory-mapped rather than held in RAM. A list of numpy views
is what hung the Jetson in notebook 09; a preallocated on-disk array cannot.
"""),
    ("code", """
def stratified_random_split(df, seed, fractions=C.SPLIT_FRACTIONS):
    out = np.empty(len(df), dtype=object)
    r = np.random.default_rng(seed)
    for lab, idx in df.groupby("label").indices.items():
        idx = np.asarray(idx).copy()
        r.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr:n_tr + n_va]] = "val"
        out[idx[n_tr + n_va:]] = "test"
    return np.asarray(out, dtype=object)


def audit_split(sub, ds):
    # Prove arm A is still leakage-safe after capping, and measure arm B.
    #
    # Capping samples rows within each (label, split) cell and never reassigns a
    # row, so it can only shrink a footprint and cannot create leakage. That
    # argument is sound, but an unverified protocol claim is exactly what this
    # paper is about, so the notebook-02 audits are repeated here on the exact
    # rows the head-to-head uses.
    rows = []

    def check(name, fn):
        try:
            fn()
            rows.append({"dataset": ds, "check": name, "result": "PASS", "detail": ""})
        except AssertionError as exc:
            rows.append({"dataset": ds, "check": name, "result": "FAIL",
                         "detail": str(exc)[:160]})

    group_key = "group_id" if ds == "tomasov" else "session_id"
    check(f"arm A: {group_key} disjoint across splits",
          lambda: S.assert_group_disjoint(sub, group_key, "split_A"))
    check("no duplicate sample_id", lambda: S.assert_no_duplicate_ids(sub))
    if ds == "tomasov":
        check("arm A: no raw-sample overlap between splits",
              lambda: S.assert_no_sample_overlap(sub, "split_A"))

    # Arm B is supposed to be contaminated.  Measure it, so the contrast rests on
    # a number rather than on the word "random".
    if ds == "tomasov":
        bpw = C.WIN_LEN // C.WIN_HOP
        hit = tot = 0
        for _, g in sub.groupby("recording_id", sort=False):
            n = int(g["block"].max()) + bpw + 1
            cov = np.zeros(n, dtype=bool)
            b_tr = g.loc[g["split_B"] == "train", "block"].to_numpy()
            for off in range(bpw):
                cov[b_tr + off] = True
            b_te = g.loc[g["split_B"] == "test", "block"].to_numpy()
            if b_te.size == 0:
                continue
            h = np.zeros(b_te.size, dtype=bool)
            for off in range(bpw):
                h |= cov[b_te + off]
            hit += int(h.sum())
            tot += int(b_te.size)
        rows.append({"dataset": ds,
                     "check": "arm B: test windows sharing raw samples with train",
                     "result": f"{100*hit/max(tot,1):.1f}%",
                     "detail": f"{hit:,} of {tot:,}"})
    else:
        spanning = int((sub.groupby("session_id")["split_B"].nunique() > 1).sum())
        total = int(sub["session_id"].nunique())
        rows.append({"dataset": ds,
                     "check": "arm B: sessions spanning train and test",
                     "result": f"{spanning}/{total}",
                     "detail": f"{100*spanning/max(total,1):.1f}% of sessions"})
    return pd.DataFrame(rows)


def capped_manifest(ds):
    mani = pd.read_parquet(C.results_dir(ds, "01_splits") / "split_manifest.parquet")
    parts = []
    for (lab, sp), g in mani.groupby(["label", "split"]):
        n = min(PER_CLASS.get(sp, 0), len(g))
        if n:
            parts.append(g.sample(n, random_state=C.SEED))
    sub = pd.concat(parts)
    if ds == "tomasov":
        sub = sub.sort_values(["h5_path", "t0", "locus"])
    else:
        sub = sub.sort_values("path")
    sub = sub.reset_index(drop=True)
    sub["split_A"] = sub["split"].to_numpy().astype(object)
    sub["split_B"] = stratified_random_split(sub, C.SEED + 404)

    audit = audit_split(sub, ds)
    display(audit)
    failed = audit[audit["result"] == "FAIL"]
    if len(failed):
        raise AssertionError(
            f"{ds}: arm A is not leakage-safe after capping - " +
            "; ".join(failed["check"] + ": " + failed["detail"]))
    return sub


def build_cache(ds, sub):
    spec = C.DATASETS[ds]
    n_nodes = NODES[ds]
    win = spec.win_len
    tag = "pp" if PREPROCESS else "raw"
    patch_path = out_dir / f"patches_{ds}_{tag}.npy"
    gb = len(sub) * n_nodes * win * 2 / 1e9
    print(f"  {ds}: {len(sub):,} windows x {n_nodes} ch x {win} -> {gb:.2f} GB float16"
          f"  ({'preprocessed' if PREPROCESS else 'raw'})")
    if gb > MAX_DISK_GB:
        raise MemoryError(f"{gb:.2f} GB exceeds MAX_DISK_GB; lower PER_CLASS.")

    if patch_path.exists() and not REBUILD_CACHE:
        P = np.load(patch_path, mmap_mode="r")
        if P.shape == (len(sub), n_nodes, win):
            print("  reusing cached patches")
            V = np.load(out_dir / f"dasvit_{ds}_{tag}.npy")
            S = np.load(out_dir / f"stats_{ds}_{tag}.npy")
            return P, V, S
        print("  cache shape mismatch, rebuilding")

    P = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.float16,
                                  shape=(len(sub), n_nodes, win))
    V = np.empty((len(sub), SO.DASVIT_DIM), dtype=np.float32)
    S = np.empty((len(sub), n_nodes, 24), dtype=np.float32)

    t0 = time.perf_counter()
    if ds == "tomasov":
        reader, cur = None, None
        try:
            for i, r in enumerate(sub.itertuples()):
                if r.h5_path != cur:
                    if reader is not None:
                        reader.close()
                    reader = pp.TomasovReader(r.h5_path)
                    cur = r.h5_path
                    print(f"    opened {Path(cur).name}", flush=True)
                patch = reader.read_patch(int(r.t0), int(r.locus), win, n_nodes)
                if PREPROCESS:
                    patch = pp.preprocess_patch(patch, reader.fs)
                P[i] = patch.astype(np.float16)
                V[i] = SO.dasvit1d_features(patch[n_nodes // 2], reader.fs)
                S[i] = SO.dastatformer_features(patch, reader.fs)
                if (i + 1) % 2000 == 0:
                    print(f"    {i+1:,}/{len(sub):,}", flush=True)
        finally:
            if reader is not None:
                reader.close()
    else:
        for i, r in enumerate(sub.itertuples()):
            patch = pp.load_cao_window(r.path, win)
            ref = pp.cao_reference_channel(patch)     # pick before z-scoring
            if PREPROCESS:
                patch = pp.preprocess_patch(patch, spec.fs)
            P[i] = patch.astype(np.float16)
            V[i] = SO.dasvit1d_features(patch[ref], spec.fs)
            S[i] = SO.dastatformer_features(patch, spec.fs)
            if (i + 1) % 2000 == 0:
                print(f"    {i+1:,}/{len(sub):,}", flush=True)

    P.flush()
    np.save(out_dir / f"dasvit_{ds}_{tag}.npy", V)
    np.save(out_dir / f"stats_{ds}_{tag}.npy", S)
    print(f"  extracted in {time.perf_counter()-t0:.0f}s")
    return np.load(patch_path, mmap_mode="r"), V, S


CACHE = {}
for ds in DATASETS:
    print(f"=== {ds} ===")
    sub = capped_manifest(ds)
    P, V, S = build_cache(ds, sub)
    CACHE[ds] = {"manifest": sub, "patch": P, "fused1d": V, "stats": S,
                 "y": sub["label"].to_numpy(),
                 "classes": sorted(sub["label"].unique())}
    print(pd.crosstab(sub["label"], sub["split_A"]))
"""),
    ("md", """
## 2. Training harness

Each model keeps its own optimiser, learning rate, batch size and epoch budget
from its source paper. Three shared choices, applied uniformly so no method is
advantaged:

* the checkpoint with the best **validation macro-F1** is the one evaluated on
  test, for every model;
* per-feature standardisation is fitted on the arm's **training rows only**, which
  DASViT1D's paper specifies and the others leave unstated;
* S2-DyGNN runs at batch 32 rather than the paper's 128, purely because its
  spectrogram tensor does not fit otherwise on the target hardware. This is
  recorded in the assumptions file.
"""),
    ("code", """
class ArrayRows(Dataset):
    # Lazy row indexing over an array or memmap.  Boolean-mask indexing would
    # copy the whole split, which for the patch cache is gigabytes.
    def __init__(self, arr, yi, idx, scale=None):
        self.arr, self.yi, self.idx, self.scale = arr, yi, idx, scale

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        j = self.idx[k]
        x = np.asarray(self.arr[j], dtype=np.float32)
        if self.scale is not None:
            x = (x - self.scale[0]) / self.scale[1]
        return torch.from_numpy(x), int(self.yi[j])


def fit_scaler(arr, idx, kind):
    # float32 mean/std over the training rows.  Computed in chunks so the patch
    # cache is never fully materialised.
    if kind == "patch":
        return None                    # S2-DyGNN normalises inside its STFT front end
    acc_n, acc_s, acc_ss = 0, None, None
    for i in range(0, len(idx), 4096):
        blk = np.asarray(arr[idx[i:i+4096]], dtype=np.float64)
        s, ss = blk.sum(axis=0), (blk ** 2).sum(axis=0)
        acc_s = s if acc_s is None else acc_s + s
        acc_ss = ss if acc_ss is None else acc_ss + ss
        acc_n += len(blk)
    mu = acc_s / acc_n
    var = np.maximum(acc_ss / acc_n - mu ** 2, 0.0)
    sd = np.sqrt(var)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


@torch.no_grad()
def predict(net, loader):
    net.eval()
    out = []
    for xb, _ in loader:
        out.append(net(xb.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=int)


def train_sota(model_name, ds, arm):
    cfg = SO.SOTA_MODELS[model_name]
    d = CACHE[ds]
    classes = d["classes"]
    arr = d[cfg["input"]]
    col = "split_A" if arm.startswith("A") else "split_B"
    sp = d["manifest"][col].to_numpy()
    yi = np.array([classes.index(v) for v in d["y"]])
    i_tr = np.flatnonzero(sp == "train")
    i_va = np.flatnonzero(sp == "val")
    i_te = np.flatnonzero(sp == "test")

    scale = fit_scaler(arr, i_tr, cfg["input"])
    if cfg["input"] == "patch":
        net = cfg["build"](len(classes), n_nodes=NODES[ds])
    elif cfg["input"] == "stats":
        net = cfg["build"](len(classes), NODES[ds])
    else:
        net = cfg["build"](len(classes))
    torch.manual_seed(C.SEED)
    net = net.to(dev)
    n_par = sum(p.numel() for p in net.parameters())

    batch = cfg["batch"]
    epochs = EPOCH_OVERRIDE or cfg["epochs"]
    opt_cls = torch.optim.AdamW if cfg["optim"] == "adamw" else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss()          # unweighted, as all three papers state

    dl_tr = DataLoader(ArrayRows(arr, yi, i_tr, scale), batch_size=batch,
                       shuffle=True, num_workers=0)
    dl_va = DataLoader(ArrayRows(arr, yi, i_va, scale), batch_size=batch * 2)
    dl_te = DataLoader(ArrayRows(arr, yi, i_te, scale), batch_size=batch * 2)

    best_f1, best_state, hist, t0 = -1.0, None, [], time.perf_counter()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            l = lossf(net(xb), yb)
            l.backward()
            if model_name == "DAStatFormer":
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(l) * len(yb)
        hist.append(tot / max(len(i_tr), 1))
        pv = predict(net, dl_va)
        m = EV.score_all(d["y"][i_va], np.array([classes[i] for i in pv]),
                         labels=classes)
        sched.step(hist[-1])
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep+1}/{epochs} loss {hist[-1]:.4f} "
                  f"val F1 {100*m['f1_macro']:.2f}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    y_pred = np.array([classes[i] for i in predict(net, dl_te)])
    y_true = d["y"][i_te]
    mt = EV.score_all(y_true, y_pred, labels=classes)
    row = {"dataset": ds, "arm": arm, "model": model_name, "n_params": n_par,
           "n_train": len(i_tr), "n_test": len(i_te),
           "val_f1_macro": 100 * best_f1,
           "accuracy": 100 * mt["accuracy"],
           "balanced_accuracy": 100 * mt["balanced_accuracy"],
           "f1_macro": 100 * mt["f1_macro"],
           "train_seconds": round(time.perf_counter() - t0, 1)}
    print(f"  {model_name} / {arm}: acc {row['accuracy']:.2f}  "
          f"F1 {row['f1_macro']:.2f}  ({n_par:,} params, {row['train_seconds']:.0f}s)")
    del net, dl_tr, dl_va, dl_te, opt
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return row, y_true, y_pred, hist
"""),
    ("md", """
## 3. Our own pipeline on exactly the same rows

The head-to-head is only meaningful if our method is evaluated on the identical
capped sample. The full-data number stays in the main results table; this one is
the matched-sample comparison.
"""),
    ("code", """
def run_ours(ds, arm):
    d = CACHE[ds]
    fused = F.load(C.results_dir(ds, "04_fusion"))
    pos = pd.Series(np.arange(len(fused["sample_id"])),
                    index=np.asarray(fused["sample_id"], dtype=str))
    ids = d["manifest"]["sample_id"].astype(str).to_numpy()
    have = np.array([i in pos.index for i in ids])
    if have.sum() < 100:
        print(f"  {ds}: only {have.sum()} matched rows in the fused matrix, skipped")
        return None
    idx = pos.loc[ids[have]].to_numpy()
    X = fused["X"][idx]
    y = d["y"][have]
    sp = d["manifest"]["split_A" if arm.startswith("A") else "split_B"].to_numpy()[have]
    tr, te = sp == "train", sp == "test"
    r = EV.fit_predict("LightGBM", "class_weight", X[tr], y[tr], X[te], y[te], C.SEED)
    m = r["metrics"]
    row = {"dataset": ds, "arm": arm, "model": "Ours (1002 features)",
           "n_params": X.shape[1], "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "val_f1_macro": np.nan,
           "accuracy": 100 * m["accuracy"],
           "balanced_accuracy": 100 * m["balanced_accuracy"],
           "f1_macro": 100 * m["f1_macro"],
           "train_seconds": round(m["train_seconds"], 1)}
    print(f"  Ours / {arm}: acc {row['accuracy']:.2f}  F1 {row['f1_macro']:.2f}")
    return row, y[te], r["y_pred"], []
"""),
    ("code", """
results, per_class, confusions, curves = [], {}, {}, {}
for ds in DATASETS:
    for arm in ARMS:
        print(f"=== {ds} / {arm} ===")
        out = run_ours(ds, arm)
        if out is not None:
            row, yt, yp, h = out
            results.append(row)
            key = (ds, arm, row["model"])
            per_class[key] = EV.per_class_report(yt, yp, labels=CACHE[ds]["classes"])
            confusions[key] = EV.confusion(yt, yp, labels=CACHE[ds]["classes"],
                                           normalize="true")
        for name in MODELS:
            try:
                row, yt, yp, h = train_sota(name, ds, arm)
            except Exception as exc:
                print(f"  {name} / {arm}: FAILED {type(exc).__name__}: {exc}")
                continue
            results.append(row)
            key = (ds, arm, name)
            per_class[key] = EV.per_class_report(yt, yp, labels=CACHE[ds]["classes"])
            confusions[key] = EV.confusion(yt, yp, labels=CACHE[ds]["classes"],
                                           normalize="true")
            curves[key] = h

res = pd.DataFrame(results)
display(res.round(2))
res.to_csv(out_dir / "head_to_head_raw.csv", index=False)
for (ds, arm, name), t in per_class.items():
    t.to_csv(out_dir / f"per_class_{ds}_{arm}_{name.replace(' ', '_').replace('/', '-')}.csv")
"""),
    ("md", """
## 4. The head-to-head table (R2C1, R1C4)

Arm A is the comparison the reviewer asked for: every method on our manifests,
our preprocessing, our metrics.
"""),
    ("code", """
head = (res[res.arm == "A_blocked"]
        .pivot_table(index=["dataset", "model"],
                     values=["accuracy", "balanced_accuracy", "f1_macro", "n_params"])
        .reset_index())
pub = PUBLISHED.rename(columns={"accuracy": "published_acc",
                                "f1_macro": "published_f1"})
head = head.merge(pub[["model", "dataset", "published_acc", "published_f1",
                       "their_protocol"]],
                  on=["model", "dataset"], how="left")
head["acc_drop_vs_published"] = (head["published_acc"] - head["accuracy"]).round(2)
head = head.sort_values(["dataset", "f1_macro"], ascending=[True, False])
display(head.round(2))
head.to_csv(out_dir / "head_to_head_under_our_protocol.csv", index=False)

infl = res.pivot_table(index=["dataset", "model"], columns="arm",
                       values="accuracy").reset_index()
if "A_blocked" in infl.columns and "B_random" in infl.columns:
    infl["protocol_inflation_pp"] = (infl["B_random"] - infl["A_blocked"]).round(2)
    display(infl.round(2))
    infl.to_csv(out_dir / "sota_protocol_inflation.csv", index=False)
    print("Inflation is how many accuracy points each method gains from a random")
    print("split on identical rows.  It is the discount that should be applied to")
    print("any published number obtained under such a split.")
"""),
    ("md", """
## 5. The openclose question (R2C2)

The decisive test. If a modern spatial-temporal architecture recovers openclose on
**our** partition, the class is learnable and our claim of a dataset limitation was
too strong. If every architecture collapses on openclose under our partition while
succeeding under a random one, the limitation is real and the published 55.7% is a
property of their partition rather than of the class.
"""),
    ("code", """
TARGET = "openclose"
oc = []
for (ds, arm, name), t in per_class.items():
    if ds != "tomasov" or TARGET not in t.index:
        continue
    oc.append({"arm": arm, "model": name,
               "openclose_f1": 100 * float(t.loc[TARGET, "f1-score"]),
               "openclose_precision": 100 * float(t.loc[TARGET, "precision"]),
               "openclose_recall": 100 * float(t.loc[TARGET, "recall"]),
               "support": int(t.loc[TARGET, "support"])})
oc = pd.DataFrame(oc).sort_values(["arm", "openclose_f1"], ascending=[True, False])
display(oc.round(2))
oc.to_csv(out_dir / "openclose_breakdown.csv", index=False)

a = oc[oc.arm == "A_blocked"]["openclose_f1"]
b = oc[oc.arm == "B_random"]["openclose_f1"]
best_a = float(a.max()) if len(a) else float("nan")
best_b = float(b.max()) if len(b) else float("nan")
pub_oc = 55.7

print(f"best openclose F1 under our leakage-safe split : {best_a:.2f}")
print(f"best openclose F1 under a random split          : {best_b:.2f}")
print(f"S2-DyGNN published (their partition)            : {pub_oc:.2f}")

if best_a >= 30.0:
    verdict = ("REPRESENTATION LIMIT - a modern architecture recovers openclose on "
               "our own partition, so the class carries usable signal and the "
               "dataset-limitation claim must be withdrawn. The correct statement "
               "is that 61 aggregate spatial scalars cannot express what a "
               "multi-channel spatial model can.")
elif best_b - best_a >= 25.0:
    verdict = ("DATASET CONFOUND - openclose is only recoverable when the partition "
               "lets near-duplicate windows straddle it. The published 55.7% is "
               "consistent with a partition effect, and our claim stands, but it "
               "must be restated as protocol-conditional rather than absolute.")
else:
    verdict = ("MIXED - neither architecture nor protocol alone accounts for the "
               "class. Report both numbers and soften the claim to a limitation of "
               "the released annotation density for this class.")
print()
print("VERDICT:", verdict)

(out_dir / "openclose_verdict.json").write_text(json.dumps({
    "best_f1_our_protocol": best_a, "best_f1_random_protocol": best_b,
    "s2dygnn_published": pub_oc, "verdict": verdict}, indent=2), encoding="utf-8")
"""),
    ("md", """
### Does openclose recovery track spatial modelling capability?

The single most informative comparison available. Under S2-DyGNN's partition the
published openclose F1 rises with each increment of inter-node modelling. If the
same ordering appears on our leakage-safe partition, the class carries real
spatial signal and our 0.044 is a limitation of aggregating spatial structure into
61 scalars. If every architecture sits near zero on our partition regardless of
capability, the published ordering was produced by the partition rather than by
the architectures.
"""),
    ("code", """
order = [m for m in SO.SPATIAL_CAPABILITY if m in set(oc.model)] if len(oc) else []
prog = []
for m in order:
    pub = SO.CMP_PUBLISHED.get(m, {})
    row = {"model": m, "published_openclose_f1": pub.get("openclose_f1", np.nan)}
    for arm in ARMS:
        sel = oc[(oc.model == m) & (oc.arm == arm)]["openclose_f1"]
        row[arm] = float(sel.iloc[0]) if len(sel) else np.nan
    prog.append(row)
prog = pd.DataFrame(prog)
if len(prog):
    display(prog.round(2))
    prog.to_csv(out_dir / "openclose_vs_spatial_capability.csv", index=False)

    from scipy.stats import spearmanr
    rank = np.arange(len(prog))
    corr = {}
    for col in ["published_openclose_f1"] + list(ARMS):
        v = prog[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        corr[col] = (float(spearmanr(rank[ok], v[ok]).statistic)
                     if ok.sum() >= 3 else float("nan"))
    print("Spearman correlation between spatial capability rank and openclose F1:")
    for k, v in corr.items():
        print(f"  {k:14s} {v:+.3f}")
    print()
    print("A strong positive correlation under A_blocked means the class responds")
    print("to spatial modelling on a leakage-safe partition, which makes our 0.044")
    print("a representation limit rather than a dataset limit.")
    (out_dir / "capability_correlation.json").write_text(
        json.dumps(corr, indent=2), encoding="utf-8")
"""),
    ("code", """
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
sub = res[(res.dataset == "tomasov")]
piv = sub.pivot_table(index="model", columns="arm", values="accuracy")
piv = piv.sort_values(piv.columns[0])
x = np.arange(len(piv))
w = 0.38
if "A_blocked" in piv.columns:
    ax.bar(x - w/2, piv["A_blocked"], w, color="#2c7fb8", label="our leakage-safe split")
if "B_random" in piv.columns:
    ax.bar(x + w/2, piv["B_random"], w, color="#fdae61", label="random split")
for i, m in enumerate(piv.index):
    p = PUBLISHED[(PUBLISHED.model == m) & (PUBLISHED.dataset == "tomasov")]
    if len(p):
        ax.plot([i - 0.45, i + 0.45], [p.accuracy.iloc[0]] * 2, "k--", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(piv.index, rotation=18, ha="right", fontsize=8)
ax.set_ylabel("test accuracy (%)")
ax.set_title("Tomasov: dashed line = each method's published figure")
ax.legend(fontsize=8)

ax = axes[1]
if len(oc):
    p2 = oc.pivot_table(index="model", columns="arm", values="openclose_f1")
    x2 = np.arange(len(p2))
    if "A_blocked" in p2.columns:
        ax.bar(x2 - w/2, p2["A_blocked"], w, color="#2c7fb8", label="leakage-safe")
    if "B_random" in p2.columns:
        ax.bar(x2 + w/2, p2["B_random"], w, color="#fdae61", label="random")
    ax.axhline(pub_oc, ls="--", color="k", lw=1.2)
    ax.text(0.02, pub_oc + 1.5, "S2-DyGNN published 55.7", fontsize=8)
    ax.set_xticks(x2)
    ax.set_xticklabels(p2.index, rotation=18, ha="right", fontsize=8)
ax.set_ylabel("openclose F1 (%)")
ax.set_title("The class R2C2 is about")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / "sota_head_to_head.png", dpi=200, bbox_inches="tight")
plt.show()
"""),
    ("md", """
## 6. What the paper must say

**R1C4 and R2C1.** Replace the CNN-only comparison with
`head_to_head_under_our_protocol.csv`. It covers a dynamic graph network, a
spectral transformer and a statistical transformer, all published in 2024 or later,
all run on our manifests with our preprocessing and our metrics. State clearly
which methods beat ours and by how much - the paper's contribution is the protocol
and the leakage evidence, and it does not require our classifier to win.

**The discount argument.** `sota_protocol_inflation.csv` gives, per method, the
accuracy gained from a random split on identical rows. That is what makes the
literature comparable: DAStatFormer's 99.93% and DASViT1D's 93.5% were obtained
under exactly that protocol, and the measured inflation is the correction.
S2-DyGNN is the exception and must be handled honestly - their five-window margin
at a 2048 shift exceeds the window length, so their partition is sample-disjoint
and their 94.0% is **not** discountable on leakage grounds.

**R2C2.** `openclose_verdict.json` decides the wording. If any architecture
recovers openclose on our own partition, the sentence claiming the class cannot be
determined must go, replaced by a limitation of our spatial representation - 61
aggregate scalars cannot express what a 17-node dynamic graph does. If none does,
the claim survives but must be restated as conditional on the protocol rather than
as a property of the data.

**Honesty about the reimplementations.** `sota_assumptions.json` lists every value
we supplied ourselves. Where a reconstruction underperforms its published figure by
more than the measured protocol inflation, say that the reconstruction is a lower
bound rather than claiming a win.
"""),
]
