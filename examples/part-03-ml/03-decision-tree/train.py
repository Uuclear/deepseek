import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
le = LabelEncoder()
y = le.fit_transform(df["species"])
X = df.drop(columns=["species"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
print("accuracy:", accuracy_score(y_te, tree.predict(X_te)))
