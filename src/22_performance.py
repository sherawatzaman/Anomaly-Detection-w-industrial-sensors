import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED, EPOCHS, BATCH = 42, 20, 128

D = ROOT / "data" / "processed"
tr, te = np.load(D / "train.npz"), np.load(D / "test.npz")
Xh3 = tr["X"][tr["y"] == 0].astype(np.float32)          # (n, 30, 15)
Xt3 = te["X"].astype(np.float32)
Xh, Xt = Xh3.mean(axis=1), Xt3.mean(axis=1)             # window-mean input


def fit_train_ae(model, Xfit, enc_fn):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.from_numpy(Xfit)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), BATCH):
            b = X[perm[i:i + BATCH]]
            opt.zero_grad()
            nn.functional.mse_loss(model(b), b).backward()
            opt.step()


class DenseAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
        self.dec = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, dim))

    def forward(self, x):
        return self.dec(self.enc(x))


class CNNAE(nn.Module):
    def __init__(self, n=15):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(n, 32, 5, padding=2), nn.ReLU(),
                                 nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(16 * 15, 8))
        self.fc = nn.Sequential(nn.Linear(8, 16 * 15), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv1d(32, n, 5, padding=2))

    def forward(self, x):                                # x: (batch, 30, sensors)
        z = self.enc(x.transpose(1, 2))
        return self.dec(self.fc(z).view(-1, 16, 15)).transpose(1, 2)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def sk(fit, score):
    return fit, score, None


specs = {}
specs["Z-score"] = (lambda: None, lambda m, S: np.abs(S).mean(axis=1), Xh, Xt, 0)
specs["PCA (k=2)"] = (lambda: PCA(2).fit(Xh),
                      lambda m, S: ((S - m.inverse_transform(m.transform(S))) ** 2).mean(axis=1), Xh, Xt, 0)
specs["Mahalanobis"] = (lambda: EmpiricalCovariance().fit(Xh), lambda m, S: m.mahalanobis(S), Xh, Xt, 0)
specs["Isolation Forest"] = (lambda: IsolationForest(n_estimators=200, random_state=SEED).fit(Xh),
                             lambda m, S: -m.score_samples(S), Xh, Xt, 0)
sub = Xh[np.random.default_rng(SEED).choice(len(Xh), 5000, replace=False)]
specs["One-Class SVM (5000 pts)"] = (lambda: OneClassSVM(nu=0.05).fit(sub),
                                     lambda m, S: -m.decision_function(S), sub, Xt, 0)


def dense_fit():
    m = DenseAE(15)
    fit_train_ae(m, Xh, None)
    return m.eval()


def cnn_fit():
    m = CNNAE(15)
    fit_train_ae(m, Xh3, None)
    return m.eval()


def torch_score(m, S):
    with torch.no_grad():
        x = torch.from_numpy(S)
        return ((m(x) - x) ** 2).mean(dim=tuple(range(1, x.ndim))).numpy()


specs[f"Dense AE ({EPOCHS} epochs)"] = (dense_fit, torch_score, Xh, Xt, "dense")
specs[f"1D-CNN AE ({EPOCHS} epochs)"] = (cnn_fit, torch_score, Xh3, Xt3, "cnn")

rows = []
for name, (fit, score, Xtrain, Xtest, kind) in specs.items():
    tracemalloc.start()
    t0 = time.perf_counter()
    model = fit()
    train_s = time.perf_counter() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    times = []
    for _ in range(3):                                   # best of 3 to reduce timing noise
        t0 = time.perf_counter()
        score(model, Xtest)
        times.append(time.perf_counter() - t0)
    rows.append({"model": name, "train_windows": len(Xtrain), "train_time_s": round(train_s, 2),
                 "peak_python_mem_MB": round(peak_mb, 1),
                 "infer_us_per_window": round(min(times) / len(Xtest) * 1e6, 2),
                 "parameters": n_params(model) if kind else 0})
    print("done", name, flush=True)

out = pd.DataFrame(rows)
out.to_csv(RES / "performance.csv", index=False)
print("\n", out.to_string(index=False))
print(f"\nTest windows scored: {len(Xt)}. Device: CPU. PyTorch {torch.__version__}")