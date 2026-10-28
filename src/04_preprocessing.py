from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cmapss"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RUL_THRESHOLD = 30      # anomalous = last 30 cycles before failure
WINDOW = 30
DROP = ["s1", "s5", "s10", "s16", "s18", "s19"]

cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=cols)

# 1. drop constant sensors
sensors = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]

# 2. RUL and labels
df["RUL"] = df.groupby("engine")["cycle"].transform("max") - df["cycle"]
df["label"] = (df["RUL"] <= RUL_THRESHOLD).astype(int)

# 3. split by engine (never by rows)
engines = df["engine"].unique()
rng = np.random.default_rng(SEED)
rng.shuffle(engines)
train_e, val_e, test_e = engines[:70], engines[70:85], engines[85:]

train = df[df["engine"].isin(train_e)]
val = df[df["engine"].isin(val_e)]
test = df[df["engine"].isin(test_e)]

# 4. scaler fitted on healthy TRAIN data only
scaler = StandardScaler().fit(train.loc[train["label"] == 0, sensors])

# 5. sliding windows
def make_windows(d):
    X, y, eng = [], [], []
    for e, g in d.groupby("engine"):
        vals = scaler.transform(g[sensors])
        labs = g["label"].values
        for i in range(WINDOW, len(g) + 1):
            X.append(vals[i - WINDOW:i])
            y.append(labs[i - 1])        # label of the last cycle in the window
            eng.append(e)
    return np.array(X), np.array(y), np.array(eng)

for name, d in [("train", train), ("val", val), ("test", test)]:
    X, y, eng = make_windows(d)
    np.savez(OUT / f"{name}.npz", X=X, y=y, engine=eng)
    print(f"{name}: windows {X.shape}, anomalous {y.mean():.1%}, engines {len(np.unique(eng))}")

# leakage check
assert not set(train_e) & set(val_e)
assert not set(train_e) & set(test_e)
assert not set(val_e) & set(test_e)
print("No engine overlap between splits.")