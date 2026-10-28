from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SKAB = ROOT / "data" / "skab"
OUT = ROOT / "data" / "processed_skab"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
WINDOW = 30                      # 30 seconds (SKAB is sampled at 1 Hz)
DROP = ["anomaly", "changepoint"]

def load(path):
    df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True).sort_index()
    return df.ffill().bfill()

# --- training data: the anomaly-free recording ---
free_files = sorted((SKAB / "anomaly-free").glob("*.csv"))
free = pd.concat([load(f) for f in free_files])
sensors = [c for c in free.columns if c not in DROP]
print("Sensors:", sensors)

scaler = StandardScaler().fit(free[sensors])     # fitted on healthy training data only

# --- labelled files, split BY FILE ---
labelled = []
for folder in ("valve1", "valve2", "other"):
    labelled += sorted((SKAB / folder).glob("*.csv"))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(labelled))
n_val = int(0.3 * len(labelled))
val_files = [labelled[i] for i in order[:n_val]]
test_files = [labelled[i] for i in order[n_val:]]

def windows_from(df, file_id, has_labels=True):
    vals = scaler.transform(df[sensors])
    labs = df["anomaly"].values.astype(int) if has_labels else np.zeros(len(df), int)
    X, y = [], []
    for i in range(WINDOW, len(df) + 1):
        X.append(vals[i - WINDOW:i])
        y.append(labs[i - 1])                    # label of the last row in the window
    return np.array(X), np.array(y), np.full(len(X), file_id)

def build(files, has_labels=True):
    parts = [windows_from(load(f), k, has_labels) for k, f in enumerate(files)]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))

splits = {"train": build(free_files, has_labels=False),
          "val": build(val_files),
          "test": build(test_files)}

for name, (X, y, fid) in splits.items():
    np.savez(OUT / f"{name}.npz", X=X, y=y, engine=fid)
    print(f"{name}: windows {X.shape}, anomalous {y.mean():.1%}, files {len(np.unique(fid))}")

assert not {f.name for f in val_files} & {f.name for f in test_files} or \
       len({str(f) for f in val_files} & {str(f) for f in test_files}) == 0
print("No file overlap between validation and test.")