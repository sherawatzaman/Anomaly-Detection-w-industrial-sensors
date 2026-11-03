from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
LATENT, EPOCHS, LR, BATCH = 8, 60, 1e-3, 128

tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
flat = lambda X: X.reshape(len(X), -1).astype(np.float32)

healthy = tr["y"] == 0
Xh, eng_h = flat(tr["X"][healthy]), tr["engine"][healthy]
Xva, Xte = flat(va["X"]), flat(te["X"])
dim = Xh.shape[1]

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, dim))
    def forward(self, x):
        return self.dec(self.enc(x))

def scores(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X)
        return ((model(x) - x) ** 2).mean(dim=1).numpy()

rows = []
for seed in SEEDS:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # hold out 10% of training ENGINES for early stopping
    engines = np.unique(eng_h)
    hold = rng.choice(engines, size=max(1, len(engines) // 10), replace=False)
    m = np.isin(eng_h, hold)
    Xfit, Xhold = torch.from_numpy(Xh[~m]), torch.from_numpy(Xh[m])

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_loss, best_state = np.inf, None

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xfit))
        for i in range(0, len(perm), BATCH):
            b = Xfit[perm[i:i + BATCH]]
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hl = loss_fn(model(Xhold), Xhold).item()
        if hl < best_loss:
            best_loss, best_state = hl, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    thr = best_threshold(va["y"], scores(model, Xva))
    res = evaluate(te["y"], scores(model, Xte), thr)
    print(f"seed {seed}: holdout loss {best_loss:.4f} | " + " ".join(f"{k} {v:.3f}" for k, v in res.items()))
    rows.append({"seed": seed, **res})

df = pd.DataFrame(rows)
df.to_csv(RES / "dense_ae_cmapss_runs.csv", index=False)
summary = df.drop(columns="seed").agg(["mean", "std"]).round(3)
print("\nDense AE, C-MAPSS, 5 seeds:")
print(summary.to_string())
summary.to_csv(RES / "dense_ae_cmapss_summary.csv")