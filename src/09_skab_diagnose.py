from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed_skab"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# Sensor order must match the "Sensors:" line printed by 07_skab_preprocessing.py
names = ["Acc1RMS", "Acc2RMS", "Current", "Pressure", "Temperature",
         "Thermocouple", "Voltage", "VolFlowRMS"]

rows = []
for split in ("val", "test"):
    d = np.load(D / f"{split}.npz")
    X, y = d["X"], d["y"]
    feats = {
        "mean":   np.abs(X.mean(axis=1)),          # level shift from healthy
        "std":    X.std(axis=1),                   # variability inside the window
        "maxabs": np.abs(X).max(axis=1),           # biggest spike in the window
        "range":  X.max(axis=1) - X.min(axis=1),   # spread inside the window
    }
    print(f"\n[{split}] windows: {len(y)}, anomalous share: {y.mean():.3f}")
    print(f"{'sensor':14s}" + "".join(f"{k:>9s}" for k in feats))
    for j, n in enumerate(names):
        aucs = {k: roc_auc_score(y, f[:, j]) for k, f in feats.items()}
        print(f"{n:14s}" + "".join(f"{v:9.3f}" for v in aucs.values()))
        rows.append({"split": split, "sensor": n, **{k: round(v, 3) for k, v in aucs.items()}})

pd.DataFrame(rows).to_csv(RES / "skab_diagnosis.csv", index=False)
print("\nSaved results/skab_diagnosis.csv")
