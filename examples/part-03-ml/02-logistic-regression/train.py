import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
le = LabelEncoder()
y = le.fit_transform(df["species"])
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("accuracy:", accuracy_score(y_te, pred))
print(classification_report(y_te, pred, target_names=le.classes_))
