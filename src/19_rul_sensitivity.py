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
RUL_THRESHOLDS = [20, 30, 50]
WINDOW = 30
DROP = ["s1", "s5", "s10", "s16", "s18", "s19"]

cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv(ROOT / "data" / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
sensors = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]
df["RUL"] = df.groupby("engine")["cycle"].transform("max") - df["cycle"]
all_eng = df["engine"].unique()

rows = []
for rul_t in RUL_THRESHOLDS:
    df["label"] = (df["RUL"] <= rul_t).astype(int)
    for seed in SPLIT_SEEDS:
        eng = np.random.default_rng(seed).permutation(all_eng)
        parts = {"train": eng[:70], "val": eng[70:85], "test": eng[85:]}
        tr_df = df[df["engine"].isin(parts["train"])]
        scaler = StandardScaler().fit(tr_df.loc[tr_df["label"] == 0, sensors])   # healthy train only

        def wins(d):
            X, y = [], []
            for _, g in d.groupby("engine"):
                X.append(sliding_window_view(scaler.transform(g[sensors]), WINDOW, axis=0).mean(-1))
                y.append(g["label"].values[WINDOW - 1:])
            return np.vstack(X), np.concatenate(y)

        Xa, ya = wins(tr_df)
        Xtr = Xa[ya == 0]
        Xva, yva = wins(df[df["engine"].isin(parts["val"])])
        Xte, yte = wins(df[df["engine"].isin(parts["test"])])

        out = {}
        z = lambda S: np.abs(S).mean(axis=1)
        out["Z-score"] = (z(Xva), z(Xte))

        best = None
        for k in (1, 2, 3, 5):
            p = PCA(n_components=k).fit(Xtr)
            err = (lambda p: lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1))(p)
            auc = roc_auc_score(yva, err(Xva))              # k chosen on validation
            if best is None or auc > best[0]:
                best = (auc, err)
        out["PCA"] = (best[1](Xva), best[1](Xte))

        emp = EmpiricalCovariance().fit(Xtr)
        out["Mahalanobis"] = (emp.mahalanobis(Xva), emp.mahalanobis(Xte))

        iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
        out["Isolation Forest"] = (-iso.score_samples(Xva), -iso.score_samples(Xte))

        idx = np.random.default_rng(seed).choice(len(Xtr), size=min(5000, len(Xtr)), replace=False)
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr[idx])
        out["One-Class SVM"] = (-oc.decision_function(Xva), -oc.decision_function(Xte))

        for name, (sv, st) in out.items():
            thr = best_threshold(yva, sv)                    # threshold from validation only
            rows.append({"rul_threshold": rul_t, "split": seed, "model": name,
                         "test_anomaly_share": yte.mean(), **evaluate(yte, st, thr)})
    print("done RUL threshold", rul_t, flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "rul_sensitivity_runs.csv", index=False)
res["rank"] = res.groupby(["rul_threshold", "split"])["roc_auc"].rank(ascending=False)

auc = res.pivot_table(index="model", columns="rul_threshold", values="roc_auc", aggfunc=["mean", "std"]).round(3)
rank = res.pivot_table(index="model", columns="rul_threshold", values="rank", aggfunc="mean").round(2)
share = res.groupby("rul_threshold")["test_anomaly_share"].mean().round(3)
auc.to_csv(RES / "rul_sensitivity_auc.csv")
rank.to_csv(RES / "rul_sensitivity_rank.csv")

print("\nMean test anomaly share:\n", share.to_string())
print("\nROC-AUC, mean and std over 10 splits:\n", auc.to_string())
print("\nMean rank (1 = best):\n", rank.to_string())