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
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SPLIT_SEEDS = list(range(42, 52))
WINDOWS = [10, 20, 30, 50]
NOISE = [0.0, 0.5, 1.0, 2.0]
BASE_WINDOW, RUL_THRESHOLD = 30, 30
DROP = ["s1", "s5", "s10", "s16", "s18", "s19"]

cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv(ROOT / "data" / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
sensors = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]
df["RUL"] = df.groupby("engine")["cycle"].transform("max") - df["cycle"]
df["label"] = (df["RUL"] <= RUL_THRESHOLD).astype(int)
all_eng = df["engine"].unique()


def wins(d, scaler, w, noise=0.0, rng=None):
    X, y = [], []
    for _, g in d.groupby("engine"):
        Z = scaler.transform(g[sensors])
        if noise > 0:
            Z = Z + rng.normal(0, noise, Z.shape)
        X.append(sliding_window_view(Z, w, axis=0).mean(-1))
        y.append(g["label"].values[w - 1:])
    return np.vstack(X), np.concatenate(y)


def fit_models(Xtr, Xva, yva, seed):
    """Return {model: score_function}, fitted on healthy training windows only."""
    f = {"Z-score": lambda S: np.abs(S).mean(axis=1)}
    best = None
    for k in (1, 2, 3, 5):
        p = PCA(n_components=k).fit(Xtr)
        err = (lambda p: lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1))(p)
        auc = roc_auc_score(yva, err(Xva))              # k chosen on validation
        if best is None or auc > best[0]:
            best = (auc, err)
    f["PCA"] = best[1]
    emp = EmpiricalCovariance().fit(Xtr)
    f["Mahalanobis"] = emp.mahalanobis
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    f["Isolation Forest"] = lambda S: -iso.score_samples(S)
    idx = np.random.default_rng(seed).choice(len(Xtr), size=min(5000, len(Xtr)), replace=False)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr[idx])
    f["One-Class SVM"] = lambda S: -oc.decision_function(S)
    return f


def split_data(seed):
    eng = np.random.default_rng(seed).permutation(all_eng)
    parts = {"train": eng[:70], "val": eng[70:85], "test": eng[85:]}
    tr_df = df[df["engine"].isin(parts["train"])]
    scaler = StandardScaler().fit(tr_df.loc[tr_df["label"] == 0, sensors])   # healthy train only
    return parts, tr_df, scaler


rows = []

# ---------------- window size ----------------
for w in WINDOWS:
    for seed in SPLIT_SEEDS:
        parts, tr_df, scaler = split_data(seed)
        Xa, ya = wins(tr_df, scaler, w)
        Xtr = Xa[ya == 0]
        Xva, yva = wins(df[df["engine"].isin(parts["val"])], scaler, w)
        Xte, yte = wins(df[df["engine"].isin(parts["test"])], scaler, w)
        for name, f in fit_models(Xtr, Xva, yva, seed).items():
            thr = best_threshold(yva, f(Xva))            # threshold from validation only
            rows.append({"experiment": "window", "value": w, "split": seed, "model": name,
                         **evaluate(yte, f(Xte), thr)})
    print("done window", w, flush=True)

# ---------------- noise on test data ----------------
for seed in SPLIT_SEEDS:
    parts, tr_df, scaler = split_data(seed)
    Xa, ya = wins(tr_df, scaler, BASE_WINDOW)
    Xtr = Xa[ya == 0]
    Xva, yva = wins(df[df["engine"].isin(parts["val"])], scaler, BASE_WINDOW)
    models = fit_models(Xtr, Xva, yva, seed)
    thrs = {n: best_threshold(yva, f(Xva)) for n, f in models.items()}
    for s in NOISE:
        rng = np.random.default_rng(1000 + seed)
        Xte, yte = wins(df[df["engine"].isin(parts["test"])], scaler, BASE_WINDOW, s, rng)
        for name, f in models.items():
            rows.append({"experiment": "noise", "value": s, "split": seed, "model": name,
                         **evaluate(yte, f(Xte), thrs[name])})
    print("done noise, split", seed, flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "ablations_runs.csv", index=False)
for exp in ("window", "noise"):
    r = res[res["experiment"] == exp]
    m = r.pivot_table(index="model", columns="value", values="roc_auc", aggfunc="mean").round(3)
    s = r.pivot_table(index="model", columns="value", values="roc_auc", aggfunc="std").round(3)
    m.to_csv(RES / f"ablation_{exp}_mean.csv")
    s.to_csv(RES / f"ablation_{exp}_std.csv")
    print(f"\n{exp.upper()}: test ROC-AUC, mean over 10 splits\n", m.to_string())
    print(f"\n{exp.upper()}: std\n", s.to_string())