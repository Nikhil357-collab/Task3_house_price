from simple_model import X_test, y_test, y_pred

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Simple Linear Regression")
print("------------------------")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R²  : {r2:.4f}")

import matplotlib.pyplot as plt

plt.scatter(
    X_test["area"],
    y_test,
    alpha=0.6,
    label="Actual"
)

plt.plot(
    X_test["area"],
    y_pred,
    label="Regression Line"
)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")

plt.legend()
plt.show()