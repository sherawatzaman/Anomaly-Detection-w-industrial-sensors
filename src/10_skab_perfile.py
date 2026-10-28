from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
SKAB = ROOT / "data" / "skab"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED, WINDOW, BASE = 42, 30, 150      # BASE = rows at the start of each file used as "healthy calibration"
DROP = ["anomaly", "changepoint"]

def load(p):
    return pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True).sort_index().ffill().bfill()

def features(path, labelled=True):
    df = load(path)
    sensors = [c for c in df.columns if c not in DROP]
    base = df[sensors].iloc[:BASE]
    if labelled and df["anomaly"].iloc[:BASE].sum() > 0:
        print("WARNING: anomaly inside baseline of", path.name)
    mu, sd = base.mean(), base.std().replace(0, 1e-6)
    Z = ((df[sensors] - mu) / sd).values                     # per-file normalisation
    W = sliding_window_view(Z, WINDOW, axis=0)               # (n, sensors, window)
    F = np.hstack([np.abs(W.mean(-1)), W.std(-1), np.abs(W).max(-1)])
    y = df["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(F), int)
    return F, y

# training: the anomaly-free recording(s)
Ftr = np.vstack([features(f, labelled=False)[0] for f in sorted((SKAB / "anomaly-free").glob("*.csv"))])

# same by-file split as 07_skab_preprocessing.py
files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
order = np.random.default_rng(SEED).permutation(len(files))
n_val = int(0.3 * len(files))
val_f = [files[i] for i in order[:n_val]]
test_f = [files[i] for i in order[n_val:]]

def build(fs):
    parts = [features(f) for f in fs]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

Fva, yva = build(val_f)
Fte, yte = build(test_f)

sc = StandardScaler().fit(Ftr)                               # fitted on healthy training data only
Xtr, Xva, Xte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)

results = {}
z = lambda S: np.abs(S).mean(axis=1)
results["Z-score"] = (z(Xva), z(Xte))

best = None
for k in (1, 2, 3, 5, 8):
    p = PCA(n_components=k).fit(Xtr)
    err = (lambda p: lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1))(p)
    auc = roc_auc_score(yva, err(Xva))
    print(f"PCA k={k} val ROC-AUC={auc:.3f}")
    if best is None or auc > best[0]:
        best = (auc, k, err)
results[f"PCA (k={best[1]})"] = (best[2](Xva), best[2](Xte))

emp = EmpiricalCovariance().fit(Xtr)
results["Mahalanobis"] = (emp.mahalanobis(Xva), emp.mahalanobis(Xte))

iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr)
results["Isolation Forest"] = (-iso.score_samples(Xva), -iso.score_samples(Xte))

idx = np.random.default_rng(SEED).choice(len(Xtr), size=min(5000, len(Xtr)), replace=False)
oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr[idx])
results["One-Class SVM"] = (-oc.decision_function(Xva), -oc.decision_function(Xte))

rows = []
for name, (sv, st) in results.items():
    thr = best_threshold(yva, sv)
    rows.append({"model": name, **evaluate(yte, st, thr)})
table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv(RES / "results_skab_perfile.csv", index=False)