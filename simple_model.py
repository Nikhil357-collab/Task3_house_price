from sklearn.model_selection import train_test_split
import pandas as pd
file_path = r"AIML_T2MODEL\data\Housing.csv"

df = pd.read_csv(file_path)


X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

# One feature
X = df[["area"]]

# Target
y = df["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
simple_model = LinearRegression()

# Train learning algorithm
simple_model.fit(X_train, y_train)

# Predict
y_pred = simple_model.predict(X_test)
print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])