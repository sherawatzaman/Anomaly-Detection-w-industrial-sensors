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
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
WINDOW, BASE, SPLIT_SEED = 30, 150, 42
EPOCHS, PATIENCE, LR, BATCH, LATENT = 40, 6, 1e-3, 256, 8
DROP = ["anomaly", "changepoint"]


class CNNAE(nn.Module):
    def __init__(self, n_sensors, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_sensors, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.ReLU(),   # 30 -> 15
            nn.Flatten(), nn.Linear(16 * 15, latent))
        self.fc = nn.Sequential(nn.Linear(latent, 16 * 15), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 15 -> 30
            nn.Conv1d(32, n_sensors, 5, padding=2))

    def encode(self, x):                      # x: (batch, 30, sensors)
        return self.enc(x.transpose(1, 2))

    def decode(self, z):
        h = self.fc(z).view(-1, 16, 15)
        return self.dec(h).transpose(1, 2)


def run(Xfit, Xhold, Xtr_all, Xva, yva, Xte, yte, n_sensors, dataset, rows):
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = CNNAE(n_sensors)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best, state, wait = np.inf, None, 0
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(Xfit))
            for i in range(0, len(perm), BATCH):
                b = Xfit[perm[i:i + BATCH]]
                opt.zero_grad()
                nn.functional.mse_loss(model.decode(model.encode(b)), b).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                hl = nn.functional.mse_loss(model.decode(model.encode(Xhold)), Xhold).item()
            if hl < best:
                best, state, wait = hl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if ep % 5 == 0:
                print(f"  {dataset} seed {seed} epoch {ep} holdout loss {hl:.4f}", flush=True)
            if wait >= PATIENCE:
                break
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            Zh = model.encode(Xtr_all).numpy()
        cov = EmpiricalCovariance().fit(Zh)            # healthy latent codes only

        def score(X, kind):
            with torch.no_grad():
                z = model.encode(X)
                if kind == "recon":
                    return ((model.decode(z) - X) ** 2).mean(dim=(1, 2)).numpy()
                return cov.mahalanobis(z.numpy())

        for kind in ("recon", "latent"):
            sv, st = score(Xva, kind), score(Xte, kind)
            thr = best_threshold(yva, sv)              # threshold from validation only
            rows.append({"dataset": dataset, "score": kind, "seed": seed,
                         "val_roc_auc": roc_auc_score(yva, sv), **evaluate(yte, st, thr)})
        print(f"DONE {dataset} seed {seed} (stopped at epoch {ep}, best loss {best:.4f})", flush=True)


rows = []

# ---------------- C-MAPSS ----------------
D = ROOT / "data" / "processed"
tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
healthy = tr["y"] == 0
Xh = tr["X"][healthy].astype(np.float32)
eng_h = tr["engine"][healthy]
rng = np.random.default_rng(42)
engines = np.unique(eng_h)
hold = rng.choice(engines, size=max(1, len(engines) // 10), replace=False)
m = np.isin(eng_h, hold)                               # hold out whole ENGINES
print("Starting C-MAPSS...", flush=True)
run(torch.from_numpy(Xh[~m]), torch.from_numpy(Xh[m]), torch.from_numpy(Xh),
    torch.from_numpy(va["X"].astype(np.float32)), va["y"],
    torch.from_numpy(te["X"].astype(np.float32)), te["y"],
    Xh.shape[2], "cmapss", rows)

# ---------------- SKAB (per-file normalisation, same split as baselines) ----------------
SKAB = ROOT / "data" / "skab"


def load(p):
    return (pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True)
            .sort_index().ffill().bfill())


def windows(path, labelled=True):
    df = load(path)
    sensors = [c for c in df.columns if c not in DROP]
    if labelled and df["anomaly"].iloc[:BASE].sum() > 0:
        return None, None                              # contaminated calibration period
    base = df[sensors].iloc[:BASE]
    mu, sd = base.mean(), base.std().replace(0, 1e-6)
    Z = ((df[sensors] - mu) / sd).values
    W = sliding_window_view(Z, WINDOW, axis=0).transpose(0, 2, 1)   # (n, 30, sensors)
    y = df["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(W), int)
    return W.astype(np.float32), y


files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
order = np.random.default_rng(SPLIT_SEED).permutation(len(files))
n_val = int(0.3 * len(files))
val_f = [files[i] for i in order[:n_val]]
test_f = [files[i] for i in order[n_val:]]
free_f = sorted((SKAB / "anomaly-free").glob("*.csv"))


def build(fs):
    parts = [p for p in (windows(f) for f in fs) if p[0] is not None]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


Wtr = np.concatenate([windows(f, labelled=False)[0] for f in free_f])
Wva, yva = build(val_f)
Wte, yte = build(test_f)

# one scaler per sensor, fitted on healthy training data only
sc = StandardScaler().fit(Wtr.reshape(-1, Wtr.shape[2]))
sc_t = lambda W: torch.from_numpy(sc.transform(W.reshape(-1, W.shape[2])).reshape(W.shape).astype(np.float32))
Xtr_all, Xva_t, Xte_t = sc_t(Wtr), sc_t(Wva), sc_t(Wte)
cut = int(0.9 * len(Xtr_all))                          # last 10% in time, with a gap
print("Starting SKAB...", flush=True)
run(Xtr_all[:cut - WINDOW], Xtr_all[cut:], Xtr_all, Xva_t, yva, Xte_t, yte,
    Wtr.shape[2], "skab", rows)

# ---------------- summary ----------------
df = pd.DataFrame(rows)
df.to_csv(RES / "cnn_ae_runs.csv", index=False)
summary = df.drop(columns="seed").groupby(["dataset", "score"]).agg(["mean", "std"]).round(3)
summary.to_csv(RES / "cnn_ae_summary.csv")
print("\n1D-CNN autoencoder, mean and std over 5 seeds:")
print(summary[["val_roc_auc", "roc_auc", "pr_auc", "f1"]].to_string())