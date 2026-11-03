from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
tr, te = np.load(D / "train.npz"), np.load(D / "test.npz")
flat = lambda X: X.reshape(len(X), -1).astype(np.float32)

Xh = torch.from_numpy(flat(tr["X"][tr["y"] == 0]))
dim = Xh.shape[1]
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 8),
                      nn.ReLU(), nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, dim))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(30):
    perm = torch.randperm(len(Xh))
    for i in range(0, len(perm), 128):
        b = Xh[perm[i:i + 128]]
        opt.zero_grad()
        nn.functional.mse_loss(model(b), b).backward()
        opt.step()

Xte = torch.from_numpy(flat(te["X"]))
with torch.no_grad():
    err = ((model(Xte) - Xte) ** 2).mean(dim=1).numpy()
mag = np.abs(te["X"]).mean(axis=(1, 2))          # how far windows sit from healthy

# remaining life of each window's last cycle (windows are stored in order per engine)
eng = te["engine"]
rul = np.zeros(len(eng), int)
for e in np.unique(eng):
    idx = np.where(eng == e)[0]
    rul[idx] = np.arange(len(idx))[::-1]

print(f"{'RUL bin':>10s} {'n':>6s} {'AE error':>10s} {'mean |z|':>10s}")
for lo, hi in [(0, 10), (10, 30), (30, 50), (50, 80), (80, 120), (120, 1000)]:
    m = (rul >= lo) & (rul < hi)
    if m.any():
        print(f"{lo:>4d}-{hi:<5d} {m.sum():>6d} {err[m].mean():>10.3f} {mag[m].mean():>10.3f}")