from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate
from sklearn.covariance import EmpiricalCovariance

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

# ---- Baseline 2: PCA reconstruction error (k chosen on validation only) ----
def make_err(p):
    return lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1)

best = None
for k in (1, 2, 3, 5, 8):
    p = PCA(n_components=k).fit(Xtr_h)
    err = make_err(p)
    auc = roc_auc_score(va["y"], err(Xva))
    print(f"PCA k = {k}  val ROC-AUC = {auc:.3f}")
    if best is None or auc > best[0]:
        best = (auc, k, err)

_, k_best, recon_err = best
print("Chosen k =", k_best, "\n")
results[f"PCA (k={k_best})"] = (recon_err(Xva), recon_err(Xte))

# ---- Baseline 3: Mahalanobis distance ----
emp = EmpiricalCovariance().fit(Xtr_h)
results["Mahalanobis"] = (emp.mahalanobis(Xva), emp.mahalanobis(Xte))

# ---- Evaluate: threshold from validation, metrics on test ----
rows = []
for name, (s_val, s_test) in results.items():
    thr = best_threshold(va["y"], s_val)
    rows.append({"model": name, **evaluate(te["y"], s_test, thr)})

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv(RES / "baseline_results_cmapss.csv", index=False)