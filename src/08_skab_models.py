from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed_skab"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42

tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
summary = lambda X: X.mean(axis=1)
Xtr, Xva, Xte = summary(tr["X"]), summary(va["X"]), summary(te["X"])

results = {}

# Z-score
z = lambda S: np.abs(S).mean(axis=1)
results["Z-score"] = (z(Xva), z(Xte))

# PCA (k chosen on validation)
best = None
for k in (1, 2, 3, 5):
    p = PCA(n_components=k).fit(Xtr)
    err = (lambda p: lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1))(p)
    auc = roc_auc_score(va["y"], err(Xva))
    print(f"PCA k = {k}  val ROC-AUC = {auc:.3f}")
    if best is None or auc > best[0]:
        best = (auc, k, err)
results[f"PCA (k={best[1]})"] = (best[2](Xva), best[2](Xte))

# Mahalanobis
emp = EmpiricalCovariance().fit(Xtr)
results["Mahalanobis"] = (emp.mahalanobis(Xva), emp.mahalanobis(Xte))

# Isolation Forest
iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr)
results["Isolation Forest"] = (-iso.score_samples(Xva), -iso.score_samples(Xte))

# One-Class SVM
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xtr), size=min(5000, len(Xtr)), replace=False)
oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr[idx])
results["One-Class SVM"] = (-oc.decision_function(Xva), -oc.decision_function(Xte))

rows = []
for name, (sv, st) in results.items():
    thr = best_threshold(va["y"], sv)           # threshold from validation only
    rows.append({"model": name, **evaluate(te["y"], st, thr)})

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv(RES / "results_skab.csv", index=False)