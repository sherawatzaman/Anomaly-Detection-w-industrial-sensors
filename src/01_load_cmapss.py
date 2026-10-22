from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent   # project folder
DATA = ROOT / "data" / "cmapss"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=cols)

print("Shape:", train.shape)                      # expect (20631, 26)
print("Missing values:", train.isna().sum().sum())
print("Engines:", train["engine"].nunique())      # expect 100
print(train.groupby("engine")["cycle"].max().describe())

for eng in [1, 2, 3, 4, 5]:
    d = train[train["engine"] == eng]
    plt.plot(d["cycle"], d["s11"], label=f"engine {eng}")
plt.xlabel("cycle")
plt.ylabel("sensor 11")
plt.legend()
plt.savefig(FIGS / "cmapss_sensor11.png", dpi=150)
plt.show()