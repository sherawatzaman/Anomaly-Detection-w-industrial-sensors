from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SPLIT_SEEDS = list(range(42, 52))
WINDOW, RUL_THRESHOLD = 30, 30
DROP = ["s1", "s5", "s10", "s16", "s18", "s19"]

cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv(ROOT / "data" / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
sensors = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]
df["RUL"] = df.groupby("engine")["cycle"].transform("max") - df["cycle"]
df["label"] = (df["RUL"] <= RUL_THRESHOLD).astype(int)
all_eng = df["engine"].unique()


def wins(d, scaler):
    X, y = [], []
    for _, g in d.groupby("engine"):
        X.append(sliding_window_view(scaler.transform(g[sensors]), WINDOW, axis=0).mean(-1))
        y.append(g["label"].values[WINDOW - 1:])
    return np.vstack(X), np.concatenate(y)


def run_models(cols_idx, Xtr, Xva, yva, Xte, yte, seed, tag, subset, rows):
    A, B, C = Xtr[:, cols_idx], Xva[:, cols_idx], Xte[:, cols_idx]
    f = {"Z-score": lambda S: np.abs(S).mean(axis=1)}
    emp = EmpiricalCovariance().fit(A)
    f["Mahalanobis"] = emp.mahalanobis
    iso = IsolationForest(n_estimators=100, random_state=seed).fit(A)
    f["Isolation Forest"] = lambda S: -iso.score_samples(S)
    for name, fn in f.items():
        thr = best_threshold(yva, fn(B))                 # threshold from validation only
        rows.append({"subset": subset, "split": seed, "model": name, "tag": tag,
                     **evaluate(yte, fn(C), thr)})


rows, loo = [], []
for seed in SPLIT_SEEDS:
    eng = np.random.default_rng(seed).permutation(all_eng)
    parts = {"train": eng[:70], "val": eng[70:85], "test": eng[85:]}
    tr_df = df[df["engine"].isin(parts["train"])]
    scaler = StandardScaler().fit(tr_df.loc[tr_df["label"] == 0, sensors])
    Xa, ya = wins(tr_df, scaler)
    Xtr = Xa[ya == 0]
    Xva, yva = wins(df[df["engine"].isin(parts["val"])], scaler)
    Xte, yte = wins(df[df["engine"].isin(parts["test"])], scaler)

    # rank sensors on VALIDATION only
    single = np.array([roc_auc_score(yva, np.abs(Xva[:, j])) for j in range(len(sensors))])
    order = np.argsort(-single)

    for k in (3, 5, 8, 15):
        run_models(order[:k], Xtr, Xva, yva, Xte, yte, seed, "top", f"top{k}", rows)
    rng = np.random.default_rng(seed)
    for r in range(5):
        run_models(rng.choice(len(sensors), 5, replace=False), Xtr, Xva, yva, Xte, yte,
                   seed * 10 + r, "random", "random5", rows)

    # leave-one-out with Z-score (validation AUC change)
    full = roc_auc_score(yva, np.abs(Xva).mean(axis=1))
    for j, s in enumerate(sensors):
        keep = [i for i in range(len(sensors)) if i != j]
        loo.append({"split": seed, "sensor": s,
                    "val_auc_drop": full - roc_auc_score(yva, np.abs(Xva[:, keep]).mean(axis=1)),
                    "single_val_auc": single[j]})
    print("done split", seed, flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "sensor_subsets_runs.csv", index=False)
m = res.groupby(["model", "subset"])["roc_auc"].agg(["mean", "std"]).round(3).unstack("subset")
m.to_csv(RES / "sensor_subsets_summary.csv")
print("\nTest ROC-AUC (mean, std over 10 splits):\n", m.to_string())

l = pd.DataFrame(loo).groupby("sensor")[["single_val_auc", "val_auc_drop"]].mean().round(3)
l = l.sort_values("single_val_auc", ascending=False)
l.to_csv(RES / "sensor_importance.csv")
print("\nSensor importance (validation):\n", l.to_string())