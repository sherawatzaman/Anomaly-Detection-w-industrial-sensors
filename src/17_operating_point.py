from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from metrics import best_threshold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED, K = 42, 3
WINDOW, BASE, RUL_LIMIT = 30, 150, 30
EPOCHS, PATIENCE, LR, BATCH = 60, 10, 1e-3, 128
DROP = ["anomaly", "changepoint"]


def sustained(flags, k=K):
    return np.convolve(flags.astype(int), np.ones(k, int), "full")[:len(flags)] >= k


def train_ae_latent(Xfit, Xhold, Xall, latent, hidden, seed):
    """Dense AE; returns a function scoring arrays by Mahalanobis distance in latent space."""
    torch.manual_seed(seed)
    dim, (h1, h2) = Xfit.shape[1], hidden
    enc = nn.Sequential(nn.Linear(dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
    dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, dim))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    Xf, Xh = torch.from_numpy(Xfit), torch.from_numpy(Xhold)
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
            state = ({k: v.clone() for k, v in enc.state_dict().items()},)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    enc.load_state_dict(state[0]); enc.eval()
    with torch.no_grad():
        cov = EmpiricalCovariance().fit(enc(torch.from_numpy(Xall)).numpy())

    def f(S):
        with torch.no_grad():
            return cov.mahalanobis(enc(torch.from_numpy(S.astype(np.float32))).numpy())
    return f


rows = []

# ======================= C-MAPSS =======================
D = ROOT / "data" / "processed"
tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
summ = lambda X: X.mean(axis=1).astype(np.float32)
healthy = tr["y"] == 0
Xh = summ(tr["X"][healthy])
eng_h = tr["engine"][healthy]
Xva, Xte = summ(va["X"]), summ(te["X"])

hold = np.random.default_rng(SEED).choice(np.unique(eng_h), size=len(np.unique(eng_h)) // 10, replace=False)
m = np.isin(eng_h, hold)                                   # hold out whole engines
iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xh)
ae = train_ae_latent(Xh[~m], Xh[m], Xh, latent=4, hidden=(32, 16), seed=SEED)
models = {"Z-score": lambda S: np.abs(S).mean(axis=1),
          "Isolation Forest": lambda S: -iso.score_samples(S),
          "Dense AE (latent)": ae}

for name, f in models.items():
    sv, st = f(Xva), f(Xte)
    thrs = {"F1": best_threshold(va["y"], sv),
            "P99": float(np.percentile(sv[va["y"] == 0], 99))}
    for tname, thr in thrs.items():
        first_rul, n_missed, n_early, n_timely, healthy_rate = [], 0, 0, 0, []
        for e in np.unique(te["engine"]):
            idx = np.where(te["engine"] == e)[0]
            rul = np.arange(len(idx))[::-1]
            mask = sustained(st[idx] >= thr)
            healthy_rate.append(mask[rul > RUL_LIMIT].mean())
            hits = np.where(mask)[0]
            if len(hits) == 0:
                n_missed += 1
                continue
            r = int(rul[hits[0]])
            first_rul.append(r)
            if r > RUL_LIMIT:
                n_early += 1
            else:
                n_timely += 1
        rows.append({"dataset": "C-MAPSS", "model": name, "threshold": tname,
                     "units": len(np.unique(te["engine"])), "timely": n_timely,
                     "early_alarm": n_early, "missed": n_missed,
                     "median_first_alarm_RUL": float(np.median(first_rul)) if first_rul else np.nan,
                     "normal_alarm_rate": float(np.mean(healthy_rate))})

# ======================= SKAB =======================
SKAB = ROOT / "data" / "skab"


def load(p):
    return (pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True)
            .sort_index().ffill().bfill())


def feats(path, labelled=True):
    df = load(path)
    sensors = [c for c in df.columns if c not in DROP]
    if labelled and df["anomaly"].iloc[:BASE].sum() > 0:
        return None
    base = df[sensors].iloc[:BASE]
    mu, sd = base.mean(), base.std().replace(0, 1e-6)
    W = sliding_window_view(((df[sensors] - mu) / sd).values, WINDOW, axis=0)
    Fm = W.mean(-1)
    Ff = W.transpose(0, 2, 1).reshape(len(W), -1)
    y = df["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(Fm), int)
    return Fm, Ff, y


files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
order = np.random.default_rng(SEED).permutation(len(files))
n_val = int(0.3 * len(files))
val_f = [files[i] for i in order[:n_val]]
test_f = [files[i] for i in order[n_val:]]

free = [feats(f, labelled=False) for f in sorted((SKAB / "anomaly-free").glob("*.csv"))]
Ftm = np.vstack([p[0] for p in free])
Ftf = np.vstack([p[1] for p in free])
sc_m, sc_f = StandardScaler().fit(Ftm), StandardScaler().fit(Ftf)    # healthy training data only
val_parts = [p for p in (feats(f) for f in val_f) if p is not None]
test_parts = [p for p in (feats(f) for f in test_f) if p is not None]

Xtr_f = sc_f.transform(Ftf).astype(np.float32)
cut = int(0.9 * len(Xtr_f))
iso2 = IsolationForest(n_estimators=200, random_state=SEED).fit(sc_m.transform(Ftm))
ae2 = train_ae_latent(Xtr_f[:cut - WINDOW], Xtr_f[cut:], Xtr_f, latent=8, hidden=(128, 32), seed=SEED)
models2 = {"Z-score": lambda Fm, Ff: np.abs(sc_m.transform(Fm)).mean(axis=1),
           "Isolation Forest": lambda Fm, Ff: -iso2.score_samples(sc_m.transform(Fm)),
           "Dense AE (latent)": lambda Fm, Ff: ae2(sc_f.transform(Ff))}

for name, f in models2.items():
    sv = np.concatenate([f(p[0], p[1]) for p in val_parts])
    yv = np.concatenate([p[2] for p in val_parts])
    thrs = {"F1": best_threshold(yv, sv),
            "P99": float(np.percentile(sv[yv == 0], 99))}
    for tname, thr in thrs.items():
        delays, missed, far = [], 0, []
        for Fm, Ff, y in test_parts:
            mask = sustained(f(Fm, Ff) >= thr)
            far.append(mask[y == 0].mean())
            if y.sum() == 0:
                continue
            onset = int(np.argmax(y == 1))
            hit = np.where(mask & (y == 1) & (np.arange(len(y)) >= onset))[0]
            if len(hit) == 0:
                missed += 1
            else:
                delays.append(int(hit[0] - onset))
        rows.append({"dataset": "SKAB", "model": name, "threshold": tname,
                     "units": len(test_parts), "timely": len(delays), "missed": missed,
                     "median_delay_s": float(np.median(delays)) if delays else np.nan,
                     "normal_alarm_rate": float(np.mean(far))})

out = pd.DataFrame(rows).round(3)
out.to_csv(RES / "operating_points.csv", index=False)
print(out.to_string(index=False))