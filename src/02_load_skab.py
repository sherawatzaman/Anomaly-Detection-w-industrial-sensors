from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
file = ROOT / "data" / "skab" / "valve1" / "0.csv"

df = pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)

print("Shape:", df.shape)
print(df.head())
print("Columns:", list(df.columns))
print("Anomaly rows:", int(df["anomaly"].sum()))