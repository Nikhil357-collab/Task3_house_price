import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==================================================
# 1. LOAD DATA
# ==================================================

file_path = "AIML_T2MODEL/data/Housing.csv"

df = pd.read_csv(file_path)


print("=" * 50)
print("VIF - MULTICOLLINEARITY ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)


# ==================================================
# 2. SEPARATE FEATURES
# ==================================================

X = df.drop("price", axis=1)


# ==================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==================================================

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


print("\nFeatures after encoding:")
print(X.shape)

print("\nFeature names:")
print(X.columns.tolist())


# ==================================================
# 4. CALCULATE VIF
# ==================================================

vif = pd.DataFrame()

vif["Feature"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(
        X.values,
        i
    )
    for i in range(X.shape[1])
]


# ==================================================
# 5. SORT VIF
# ==================================================

vif = vif.sort_values(
    by="VIF",
    ascending=False
)


# ==================================================
# 6. DISPLAY RESULT
# ==================================================

print("\n" + "=" * 50)
print("VIF RESULTS")
print("=" * 50)

print(
    vif.to_string(index=False)
)