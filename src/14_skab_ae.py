from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
SKAB = ROOT / "data" / "skab"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
SPLIT_SEED, WINDOW, BASE = 42, 30, 150
EPOCHS, LR, BATCH = 60, 1e-3, 128
DROP = ["anomaly", "changepoint"]
CONFIG = {"flat": dict(latent=8, hidden=(128, 32)),     # 30 x 8 = 240 inputs
          "mean": dict(latent=4, hidden=(32, 16))}      # 8 inputs


def load(p):
    return (pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True)
            .sort_index().ffill().bfill())


def features(path, variant, labelled=True):
    df = load(path)
    sensors = [c for c in df.columns if c not in DROP]
    if labelled and df["anomaly"].iloc[:BASE].sum() > 0:
        return None, None                                # contaminated calibration period
    base = df[sensors].iloc[:BASE]
    mu, sd = base.mean(), base.std().replace(0, 1e-6)
    Z = ((df[sensors] - mu) / sd).values                 # per-file normalisation
    W = sliding_window_view(Z, WINDOW, axis=0)           # (n, sensors, window)
    F = W.mean(-1) if variant == "mean" else W.transpose(0, 2, 1).reshape(len(W), -1)
    y = df["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(F), int)
    return F.astype(np.float32), y


# same by-file split as the SKAB baselines
files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
order = np.random.default_rng(SPLIT_SEED).permutation(len(files))
n_val = int(0.3 * len(files))
val_f = [files[i] for i in order[:n_val]]
test_f = [files[i] for i in order[n_val:]]
free_f = sorted((SKAB / "anomaly-free").glob("*.csv"))


def build(fs, variant):
    parts = [features(f, variant) for f in fs]
    parts = [p for p in parts if p[0] is not None]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def make_ae(dim, latent, hidden):
    h1, h2 = hidden
    enc = nn.Sequential(nn.Linear(dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
    dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, dim))
    return enc, dec


rows = []
for variant in ("flat", "mean"):
    cfg = CONFIG[variant]
    Ftr = np.vstack([features(f, variant, labelled=False)[0] for f in free_f])
    Fva, yva = build(val_f, variant)
    Fte, yte = build(test_f, variant)
    sc = StandardScaler().fit(Ftr)                       # fitted on healthy training data only
    Xtr = sc.transform(Ftr).astype(np.float32)
    Xva = torch.from_numpy(sc.transform(Fva).astype(np.float32))
    Xte = torch.from_numpy(sc.transform(Fte).astype(np.float32))

    # early stopping: last 10% of the healthy recording (in time), with a gap so windows don't overlap
    cut = int(0.9 * len(Xtr))
    Xfit = torch.from_numpy(Xtr[:cut - WINDOW])
    Xhold = torch.from_numpy(Xtr[cut:])

    for seed in SEEDS:
        torch.manual_seed(seed)
        enc, dec = make_ae(Xtr.shape[1], cfg["latent"], cfg["hidden"])
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
        best, state = np.inf, None
        for ep in range(EPOCHS):
            enc.train(); dec.train()
            perm = torch.randperm(len(Xfit))
            for i in range(0, len(perm), BATCH):
                b = Xfit[perm[i:i + BATCH]]
                opt.zero_grad()
                nn.functional.mse_loss(dec(enc(b)), b).backward()
                opt.step()
            enc.eval(); dec.eval()
            with torch.no_grad():
                hl = nn.functional.mse_loss(dec(enc(Xhold)), Xhold).item()
            if hl < best:
                best = hl
                state = ({k: v.clone() for k, v in enc.state_dict().items()},
                         {k: v.clone() for k, v in dec.state_dict().items()})
        enc.load_state_dict(state[0]); dec.load_state_dict(state[1])
        enc.eval(); dec.eval()

        with torch.no_grad():
            Zh = enc(torch.from_numpy(Xtr)).numpy()
        cov = EmpiricalCovariance().fit(Zh)              # healthy latent codes only

        def score(X, kind):
            with torch.no_grad():
                z = enc(X)
                if kind == "recon":
                    return ((dec(z) - X) ** 2).mean(dim=1).numpy()
                return cov.mahalanobis(z.numpy())

        for kind in ("recon", "latent"):
            sv, st = score(Xva, kind), score(Xte, kind)
            thr = best_threshold(yva, sv)                # threshold from validation only
            res = evaluate(yte, st, thr)
            rows.append({"variant": variant, "score": kind, "seed": seed,
                         "val_roc_auc": roc_auc_score(yva, sv), **res})
        print("done", variant, seed)

df = pd.DataFrame(rows)
df.to_csv(RES / "ae_skab_runs.csv", index=False)
summary = df.drop(columns="seed").groupby(["variant", "score"]).agg(["mean", "std"]).round(3)
summary.to_csv(RES / "ae_skab_summary.csv")
print("\nSKAB dense AE, mean and std over 5 seeds:")
print(summary[["val_roc_auc", "roc_auc", "pr_auc", "f1"]].to_string())