# Dataset

The raw data is **not tracked in git**: about 47 GB, redistributed by its
authors under their own terms. Download both datasets and place them here:

```text
Dataset/
├── Cao_2023/
│   ├── Training/01_background … 06_walk/*.mat   12,335 records (10000 × 12, uint16)
│   ├── Test/01_background … 06_walk/*.mat        3,084 records
│   └── Phi-OTDR_dataset_and_codes-main/          the authors' code (CNN baseline, notebook 10)
└── Tomasov_2024/
    └── data/<class>/<recording>.h5 · .json · .npy   9 classes, 16 recordings
```

| Folder | Dataset | Obtain from |
|---|---|---|
| `Cao_2023/` | Cao, Su, Jin & Yu, *An Open Dataset of Φ-OTDR Events with Two Classification Models as Baselines*, Results in Optics 10:100372 (2023) | <https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes> (its README links the Google Drive / Baidu copies of the data) |
| `Tomasov_2024/` | Tomasov et al., *Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems*, Scientific Data 12:793 (2025) | the data record linked from the paper |

## Keeping the data somewhere else

`Experiment/dasfe/config.py` looks for the data here by default. To use
another location without moving 47 GB (an external drive, or the Jetson's
existing mount), point an environment variable at it:

```bash
export DAS_DATASET_DIR=/mnt/<disk>/Paper_Reeyan/Dataset          # Linux / Jetson
```

```powershell
$env:DAS_DATASET_DIR = "D:\Paper_Reeyan\Dataset"                  # Windows
```

`DAS_RESULTS_DIR` does the same for `Results/`.

The frozen manifests in `Results/` store absolute paths from the machine that
built them (the Jetson). `config.data_path()` re-roots those paths onto this
folder at load time, so the manifests never need rebuilding when the data moves.
