"""Health check for the experiment environment.  Writes nothing.

    python scripts/check_env.py               # report
    python scripts/check_env.py --require-gpu # also fail if PyTorch cannot use CUDA

Reports the interpreter and package versions (the same list notebook 00
checks), whether PyTorch can run on the GPU, whether `dasfe` imports with its
1,002 features, and whether the stored manifests' raw-data paths resolve on
this machine.  Exits non-zero if anything required is missing.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Experiment"))

# import name -> pip name.  Same REQUIRED list as notebook 00.
REQUIRED = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "h5py": "h5py", "pywt": "PyWavelets",
    "skimage": "scikit-image", "joblib": "joblib", "tqdm": "tqdm", "pyarrow": "pyarrow",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "imblearn": "imbalanced-learn",
    "shap": "shap", "boruta": "Boruta", "skrebate": "skrebate", "mrmr": "mrmr-selection",
}
OPTIONAL = {"torch": "torch", "librosa": "librosa"}


def check_packages() -> bool:
    ok = True
    for mod, pip_name in {**REQUIRED, **OPTIONAL}.items():
        try:
            version = getattr(importlib.import_module(mod), "__version__", "n/a")
        except ImportError:
            ok &= mod not in REQUIRED
            version = "MISSING" + ("   <-- required" if mod in REQUIRED else " (optional)")
        print(f"  {pip_name:18s} {version}")
    return ok


def check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print("gpu      PyTorch not installed in this environment")
        return False
    if not torch.cuda.is_available():
        print(f"gpu      torch {torch.__version__} cannot see a CUDA device (CPU only)")
        return False
    x = torch.randn(2048, 2048, device="cuda")
    (x @ x).sum().item()                                   # forces a real kernel launch
    print(f"gpu      {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | matmul OK")
    return True


def check_dasfe() -> bool:
    try:
        import dasfe
        from dasfe import config as C
    except Exception as exc:                                # missing deps or a drifted extractor
        print(f"dasfe    FAILED to import: {exc!r}")
        return False
    print(f"dasfe    {dasfe.__version__} | {dasfe.TOTAL_FEATURES} features | n_jobs={C.N_JOBS}")
    for name, d in (("Dataset", C.DATASET_DIR), ("Cao_2023", C.CAO_DIR),
                    ("Tomasov", C.TOMASOV_DIR), ("Results", C.RESULTS_DIR)):
        print(f"  {name:9s} {'ok       ' if d.is_dir() else 'NOT FOUND'} {d}")

    # The frozen manifests carry absolute paths from the machine that built them;
    # config.data_path() must map them onto this machine's Dataset/.
    import pandas as pd
    for ds, col in (("cao", "path"), ("tomasov", "h5_path")):
        mani = C.RESULTS_DIR / ds / "01_splits" / "split_manifest.parquet"
        if not mani.exists():
            continue
        paths = pd.read_parquet(mani, columns=[col])[col].drop_duplicates()
        found = sum(C.data_path(p).exists() for p in paths)
        print(f"  manifest {ds:8s} {found}/{len(paths)} raw files resolve on this machine")
    return True


def main(argv: list[str]) -> int:
    print(f"python   {platform.python_version()}  ({sys.prefix})")
    print(f"platform {platform.platform()}")
    ok = check_packages()
    gpu = check_gpu()
    ok &= check_dasfe()
    if "--require-gpu" in argv and not gpu:
        ok = False
    print("\nOK" if ok else "\nPROBLEMS FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
