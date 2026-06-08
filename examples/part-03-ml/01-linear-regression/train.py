import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
X = df[["petal_length"]].values
y = df["sepal_length"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("coef:", model.coef_[0], "intercept:", model.intercept_)
print("RMSE:", mean_squared_error(y_te, pred, squared=False).__round__(3))
print("R2:", r2_score(y_te, pred).__round__(3))
