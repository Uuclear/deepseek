import pandas as pd
from pathlib import Path
csv = Path(__file__).resolve().parents[3] / "data" / "iris.csv"
df = pd.read_csv(csv)
print(df.head())
print("\n按 species 计数:\n", df["species"].value_counts())
print("\n数值列均值:\n", df.select_dtypes("number").mean())
