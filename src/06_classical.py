from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42

tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))

def summary(X):
    return X.mean(axis=1)               # one 15-number summary per window

Xtr_h = summary(tr["X"][tr["y"] == 0])  # healthy training windows only
Xva, Xte = summary(va["X"]), summary(te["X"])

results = {}

# ---- Isolation Forest ----
iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr_h)
results["Isolation Forest"] = (-iso.score_samples(Xva), -iso.score_samples(Xte))

# ---- One-Class SVM (subsample healthy data to keep it fast) ----
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xtr_h), size=min(5000, len(Xtr_h)), replace=False)
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr_h[idx])
results["One-Class SVM"] = (-ocsvm.decision_function(Xva), -ocsvm.decision_function(Xte))

rows = []
for name, (s_val, s_test) in results.items():
    thr = best_threshold(va["y"], s_val)   # threshold from validation only
    rows.append({"model": name, **evaluate(te["y"], s_test, thr)})

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv(RES / "classical_results_cmapss.csv", index=False)