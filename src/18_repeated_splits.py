from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

SPLIT_SEEDS = list(range(42, 52))          # 10 different splits
WINDOW, BASE, RUL_THRESHOLD = 30, 150, 30
EPOCHS, PATIENCE, LR, BATCH = 60, 10, 1e-3, 128
DROP_C = ["s1", "s5", "s10", "s16", "s18", "s19"]
DROP_S = ["anomaly", "changepoint"]


# ---------------- models ----------------
def ae_latent(Xtr, fit_idx, hold_idx, seed):
    torch.manual_seed(seed)
    Xf = torch.from_numpy(Xtr[fit_idx].astype(np.float32))
    Xh = torch.from_numpy(Xtr[hold_idx].astype(np.float32))
    dim = Xtr.shape[1]
    enc = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    dec = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, dim))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    best, state, wait = np.inf, None, 0
    for ep in range(EPOCHS):
        enc.train(); dec.train()
        perm = torch.randperm(len(Xf))
        for i in range(0, len(perm), BATCH):
            b = Xf[perm[i:i + BATCH]]
            opt.zero_grad()
            nn.functional.mse_loss(dec(enc(b)), b).backward()
            opt.step()
        enc.eval(); dec.eval()
        with torch.no_grad():
            hl = nn.functional.mse_loss(dec(enc(Xh)), Xh).item()
        if hl < best:
            best, wait = hl, 0
            state = ({k: v.clone() for k, v in enc.state_dict().items()},
                     {k: v.clone() for k, v in dec.state_dict().items()})
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    enc.load_state_dict(state[0]); enc.eval()
    with torch.no_grad():
        cov = EmpiricalCovariance().fit(enc(torch.from_numpy(Xtr.astype(np.float32))).numpy())

    def f(S):
        with torch.no_grad():
            return cov.mahalanobis(enc(torch.from_numpy(S.astype(np.float32))).numpy())
    return f


def all_scores(Xtr, Xva, yva, Xte, seed, fit_idx, hold_idx):
    """Return {model: (val_scores, test_scores)}. Everything is fitted on healthy training data only."""
    out = {}
    z = lambda S: np.abs(S).mean(axis=1)
    out["Z-score"] = (z(Xva), z(Xte))

    best = None
    for k in (1, 2, 3, 5):
        if k > Xtr.shape[1]:
            continue
        p = PCA(n_components=k).fit(Xtr)
        err = (lambda p: lambda S: ((S - p.inverse_transform(p.transform(S))) ** 2).mean(axis=1))(p)
        auc = roc_auc_score(yva, err(Xva))             # k chosen on validation
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

    ae = ae_latent(Xtr, fit_idx, hold_idx, seed)
    out["Dense AE (latent)"] = (ae(Xva), ae(Xte))
    return out


rows = []


def record(dataset, split, scores, yva, yte):
    for name, (sv, st) in scores.items():
        thr = best_threshold(yva, sv)                  # threshold from validation only
        rows.append({"dataset": dataset, "split": split, "model": name, **evaluate(yte, st, thr)})


# ======================= C-MAPSS: 10 engine splits =======================
cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv(ROOT / "data" / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
sensors_c = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP_C]
df["RUL"] = df.groupby("engine")["cycle"].transform("max") - df["cycle"]
df["label"] = (df["RUL"] <= RUL_THRESHOLD).astype(int)
all_eng = df["engine"].unique()

for seed in SPLIT_SEEDS:
    eng = np.random.default_rng(seed).permutation(all_eng)
    parts = {"train": eng[:70], "val": eng[70:85], "test": eng[85:]}
    tr_df = df[df["engine"].isin(parts["train"])]
    scaler = StandardScaler().fit(tr_df.loc[tr_df["label"] == 0, sensors_c])   # healthy train only

    def wins(d):
        X, y, e = [], [], []
        for k, g in d.groupby("engine"):
            W = sliding_window_view(scaler.transform(g[sensors_c]), WINDOW, axis=0).mean(-1)
            X.append(W); y.append(g["label"].values[WINDOW - 1:]); e.append(np.full(len(W), k))
        return np.vstack(X), np.concatenate(y), np.concatenate(e)

    Xa, ya, ea = wins(tr_df)
    Xtr, etr = Xa[ya == 0], ea[ya == 0]
    Xva, yva, _ = wins(df[df["engine"].isin(parts["val"])])
    Xte, yte, _ = wins(df[df["engine"].isin(parts["test"])])
    uniq = np.unique(etr)
    hold = np.random.default_rng(seed).choice(uniq, size=len(uniq) // 10, replace=False)
    m = np.isin(etr, hold)                              # hold out whole engines
    sc = all_scores(Xtr, Xva, yva, Xte, seed, np.where(~m)[0], np.where(m)[0])
    record("C-MAPSS", seed, sc, yva, yte)
    print("done C-MAPSS split", seed, flush=True)

# ======================= SKAB: 10 file splits =======================
SKAB = ROOT / "data" / "skab"
cache = {}


def feats(path, labelled=True):
    key = (str(path), labelled)
    if key in cache:
        return cache[key]
    d = (pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True).sort_index().ffill().bfill())
    s = [c for c in d.columns if c not in DROP_S]
    if labelled and d["anomaly"].iloc[:BASE].sum() > 0:
        res = (None, None)                              # contaminated calibration period
    else:
        base = d[s].iloc[:BASE]
        mu, sd = base.mean(), base.std().replace(0, 1e-6)
        F = sliding_window_view(((d[s] - mu) / sd).values, WINDOW, axis=0).mean(-1)
        y = d["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(F), int)
        res = (F, y)
    cache[key] = res
    return res


files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
Ftr = np.vstack([feats(f, labelled=False)[0] for f in sorted((SKAB / "anomaly-free").glob("*.csv"))])
sc_s = StandardScaler().fit(Ftr)                        # healthy training data only
Xtr = sc_s.transform(Ftr)
cut = int(0.9 * len(Xtr))                               # last 10% in time for early stopping, with a gap
fit_idx, hold_idx = np.arange(cut - WINDOW), np.arange(cut, len(Xtr))


def build(fs):
    p = [feats(f) for f in fs]
    p = [q for q in p if q[0] is not None]
    return sc_s.transform(np.vstack([q[0] for q in p])), np.concatenate([q[1] for q in p])


for seed in SPLIT_SEEDS:
    order = np.random.default_rng(seed).permutation(len(files))
    n_val = int(0.3 * len(files))
    Xva, yva = build([files[i] for i in order[:n_val]])
    Xte, yte = build([files[i] for i in order[n_val:]])
    sc = all_scores(Xtr, Xva, yva, Xte, seed, fit_idx, hold_idx)
    record("SKAB", seed, sc, yva, yte)
    print("done SKAB split", seed, flush=True)

# ======================= summary =======================
res = pd.DataFrame(rows)
res.to_csv(RES / "repeated_splits_runs.csv", index=False)
res["rank"] = res.groupby(["dataset", "split"])["roc_auc"].rank(ascending=False)
g = res.groupby(["dataset", "model"])
summary = g[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"]).round(3)
summary[("rank", "mean")] = g["rank"].mean().round(2)
summary = summary.sort_values(["dataset", ("roc_auc", "mean")], ascending=[True, False])
summary.to_csv(RES / "repeated_splits_summary.csv")
print("\nMean and std over", len(SPLIT_SEEDS), "splits (test):")
print(summary.to_string())