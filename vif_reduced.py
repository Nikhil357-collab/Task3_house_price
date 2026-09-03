import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. LOAD DATA
# ==================================================

file_path = "AIML_T2MODEL/data/Housing.csv"

df = pd.read_csv(file_path)


# ==================================================
# 2. SEPARATE X AND y
# ==================================================

X = df.drop("price", axis=1)
y = df["price"]


# ==================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==================================================

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


# ==================================================
# 4. REMOVE HIGH-VIF FEATURE
# ==================================================

X = X.drop(
    "bedrooms",
    axis=1
)


print("Features used:")
print(X.columns.tolist())


# ==================================================
# 5. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 6. TRAIN MODEL
# ==================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==================================================
# 7. PREDICTION
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 8. EVALUATION
# ==================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ==================================================
# 9. RESULTS
# ==================================================

print("\n" + "=" * 50)
print("REDUCED MULTIPLE LINEAR REGRESSION")
print("=" * 50)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")