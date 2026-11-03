from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from metrics import best_threshold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED, K = 42, 3                     # K = consecutive flagged windows needed for an alarm
WINDOW, BASE, RUL_LIMIT = 30, 150, 30
DROP = ["anomaly", "changepoint"]


def sustained(flags, k=K):
    """True where the last k windows were all flagged."""
    return np.convolve(flags.astype(int), np.ones(k, int), "full")[:len(flags)] >= k


rows = []

# ======================= C-MAPSS: early warning lead time =======================
D = ROOT / "data" / "processed"
tr, va, te = (np.load(D / f"{n}.npz") for n in ("train", "val", "test"))
summ = lambda X: X.mean(axis=1)
Xh = summ(tr["X"][tr["y"] == 0])
Xva, Xte = summ(va["X"]), summ(te["X"])
iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xh)
models = {"Z-score": lambda S: np.abs(S).mean(axis=1),
          "Isolation Forest": lambda S: -iso.score_samples(S)}

for name, f in models.items():
    thr = best_threshold(va["y"], f(Xva))
    s = f(Xte)
    leads, n_missed, n_false = [], 0, 0
    healthy_alarm = []
    engines = np.unique(te["engine"])
    for e in engines:
        idx = np.where(te["engine"] == e)[0]
        n = len(idx)
        rul = np.arange(n)[::-1]                       # windows are stored in time order
        mask = sustained(s[idx] >= thr)
        healthy_alarm.append(mask[rul > RUL_LIMIT].mean())
        hits = np.where(mask)[0]
        if len(hits) == 0:
            n_missed += 1
        elif rul[hits[0]] > RUL_LIMIT:
            n_false += 1                               # first alarm while still healthy
        else:
            leads.append(int(rul[hits[0]]))
    rows.append({"dataset": "C-MAPSS", "model": name, "units": len(engines),
                 "timely": len(leads), "early_false_alarm": n_false, "missed": n_missed,
                 "median_lead_cycles": float(np.median(leads)) if leads else np.nan,
                 "healthy_alarm_rate": float(np.mean(healthy_alarm))})

# ======================= SKAB: detection delay and false alarms =======================
SKAB = ROOT / "data" / "skab"


def load(p):
    return (pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True)
            .sort_index().ffill().bfill())


def feats(path, labelled=True):
    df = load(path)
    sensors = [c for c in df.columns if c not in DROP]
    if labelled and df["anomaly"].iloc[:BASE].sum() > 0:
        return None, None                              # contaminated calibration period
    base = df[sensors].iloc[:BASE]
    mu, sd = base.mean(), base.std().replace(0, 1e-6)
    Z = ((df[sensors] - mu) / sd).values
    F = sliding_window_view(Z, WINDOW, axis=0).mean(-1)
    y = df["anomaly"].values[WINDOW - 1:].astype(int) if labelled else np.zeros(len(F), int)
    return F, y


files = []
for folder in ("valve1", "valve2", "other"):
    files += sorted((SKAB / folder).glob("*.csv"))
order = np.random.default_rng(SEED).permutation(len(files))
n_val = int(0.3 * len(files))
val_f = [files[i] for i in order[:n_val]]
test_f = [files[i] for i in order[n_val:]]

Ftr = np.vstack([feats(f, labelled=False)[0] for f in sorted((SKAB / "anomaly-free").glob("*.csv"))])
sc = StandardScaler().fit(Ftr)
va_parts = [p for p in (feats(f) for f in val_f) if p[0] is not None]
Xva_s = sc.transform(np.vstack([p[0] for p in va_parts]))
yva_s = np.concatenate([p[1] for p in va_parts])
test_parts = [p for p in (feats(f) for f in test_f) if p[0] is not None]
iso2 = IsolationForest(n_estimators=200, random_state=SEED).fit(sc.transform(Ftr))
models2 = {"Z-score": lambda S: np.abs(S).mean(axis=1),
           "Isolation Forest": lambda S: -iso2.score_samples(S)}

for name, f in models2.items():
    thr = best_threshold(yva_s, f(Xva_s))
    delays, missed, far = [], 0, []
    for F, y in test_parts:
        mask = sustained(f(sc.transform(F)) >= thr)
        far.append(mask[y == 0].mean())
        if y.sum() == 0:
            continue
        onset = int(np.argmax(y == 1))
        hit = np.where(mask & (y == 1) & (np.arange(len(y)) >= onset))[0]
        if len(hit) == 0:
            missed += 1
        else:
            delays.append(int(hit[0] - onset))         # rows = seconds (1 Hz)
    rows.append({"dataset": "SKAB", "model": name, "units": len(test_parts),
                 "timely": len(delays), "early_false_alarm": np.nan, "missed": missed,
                 "median_delay_s": float(np.median(delays)) if delays else np.nan,
                 "false_alarm_rate": float(np.mean(far))})

out = pd.DataFrame(rows).round(3)
out.to_csv(RES / "early_warning.csv", index=False)
print(out.to_string(index=False))