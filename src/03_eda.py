from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures without opening windows
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ---------- C-MAPSS ----------
cols = ["engine", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv(ROOT / "data" / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)

sensors = [f"s{i}" for i in range(1, 22)]
std = train[sensors].std()
constant = std[std < 1e-6].index.tolist()
print("Constant sensors (drop these):", constant)
print("\nLowest-variance sensors:\n", std.sort_values().head(8))

keep = [s for s in sensors if s not in constant]
corr = train[keep].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="correlation")
plt.xticks(range(len(keep)), keep, rotation=90)
plt.yticks(range(len(keep)), keep)
plt.title("C-MAPSS FD001 sensor correlations")
plt.tight_layout()
plt.savefig(FIGS / "cmapss_correlation.png", dpi=150)
plt.close()

# ---------- SKAB ----------
df = pd.read_csv(ROOT / "data" / "skab" / "valve1" / "0.csv",
                 sep=";", index_col="datetime", parse_dates=True)
sensor_cols = [c for c in df.columns if c not in ("anomaly", "changepoint")]

fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(12, 14), sharex=True)
for ax, c in zip(axes, sensor_cols):
    ax.plot(df.index, df[c], linewidth=0.8)
    ax.fill_between(df.index, df[c].min(), df[c].max(),
                    where=df["anomaly"] == 1, color="red", alpha=0.25)
    ax.set_ylabel(c, fontsize=7)
plt.tight_layout()
plt.savefig(FIGS / "skab_anomalies.png", dpi=150)
plt.close()

print("\nSaved figures to", FIGS)