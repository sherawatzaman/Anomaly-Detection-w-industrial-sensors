from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))

# one 15-number summary per window (mean over the 30 timesteps)
def summary(X):
    return X.mean(axis=1)

Xtr_h = summary(tr["X"][tr["y"] == 0])     # healthy training windows only
Xva, Xte = summary(va["X"]), summary(te["X"])

results = {}

# ---- Baseline 1: Z-score (mean absolute z over sensors) ----
z = lambda S: np.abs(S).mean(axis=1)       # data already standardised on healthy train
results["Z-score"] = (z(Xva), z(Xte))

# ---- Baseline 2: PCA reconstruction error ----
pca = PCA(n_components=5).fit(Xtr_h)
def recon_err(S):
    rec = pca.inverse_transform(pca.transform(S))
    return ((S - rec) ** 2).mean(axis=1)
results["PCA"] = (recon_err(Xva), recon_err(Xte))

rows = []
for name, (s_val, s_test) in results.items():
    thr = best_threshold(va["y"], s_val)    # threshold from validation only
    m = evaluate(te["y"], s_test, thr)
    rows.append({"model": name, **m})

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv(RES / "baseline_results_cmapss.csv", index=False)