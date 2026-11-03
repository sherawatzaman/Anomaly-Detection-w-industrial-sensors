from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
EPOCHS, LR, BATCH = 60, 1e-3, 128

tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
healthy = tr["y"] == 0
eng_h = tr["engine"][healthy]

def prep(X, variant):
    X = X.reshape(len(X), -1) if variant == "flat" else X.mean(axis=1)
    return X.astype(np.float32)

CONFIG = {"flat": dict(latent=8, hidden=(128, 32)),
          "mean": dict(latent=4, hidden=(32, 16))}

def make_ae(dim, latent, hidden):
    h1, h2 = hidden
    enc = nn.Sequential(nn.Linear(dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
    dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, dim))
    return enc, dec

def train(variant, seed):
    cfg = CONFIG[variant]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xh = prep(tr["X"][healthy], variant)
    engines = np.unique(eng_h)
    hold = rng.choice(engines, size=max(1, len(engines) // 10), replace=False)
    m = np.isin(eng_h, hold)                      # hold out whole ENGINES for early stopping
    Xfit, Xhold = torch.from_numpy(Xh[~m]), torch.from_numpy(Xh[m])

    enc, dec = make_ae(Xh.shape[1], cfg["latent"], cfg["hidden"])
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=LR)
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
        Zh = enc(torch.from_numpy(Xh)).numpy()
    cov = EmpiricalCovariance().fit(Zh)           # healthy latent codes only

    def score(X, kind):
        x = torch.from_numpy(prep(X, variant))
        with torch.no_grad():
            z = enc(x)
            if kind == "recon":
                return ((dec(z) - x) ** 2).mean(dim=1).numpy()
            return cov.mahalanobis(z.numpy())
    return score

rows = []
for variant in ("flat", "mean"):
    for seed in SEEDS:
        score = train(variant, seed)
        for kind in ("recon", "latent"):
            sv, st = score(va["X"], kind), score(te["X"], kind)
            thr = best_threshold(va["y"], sv)     # threshold from validation only
            res = evaluate(te["y"], st, thr)
            rows.append({"variant": variant, "score": kind, "seed": seed,
                         "val_roc_auc": roc_auc_score(va["y"], sv), **res})
        print("done", variant, seed)

df = pd.DataFrame(rows)
df.to_csv(RES / "dense_ae_variants_cmapss_runs.csv", index=False)
g = df.drop(columns="seed").groupby(["variant", "score"])
summary = g.agg(["mean", "std"]).round(3)
summary.to_csv(RES / "dense_ae_variants_cmapss_summary.csv")
print("\nTest metrics, mean and std over 5 seeds:")
print(summary[["val_roc_auc", "roc_auc", "pr_auc", "f1"]].to_string())