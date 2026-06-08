import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "titanic_sample.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = (df["Sex"] == "female").astype(int)
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500)),
])
scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
print("CV accuracy:", scores.round(3), "mean:", scores.mean().round(3))
