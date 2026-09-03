from multiple_regre import load_and_preprocess_data
from multiple_train import train_linear_regression, display_coefficients
from m_evaluate import evaluate_model, plot_predictions
from diagonistic import plot_residuals, residual_summary
from cross_val import cross_validate
from statsmodels.stats.diagnostic import het_breuschpagan
# Dataset path

file_path = r"AIML_T2MODEL\data\Housing.csv"


# 1. Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(
    file_path
)


# 2. Train Linear Regression model
model = train_linear_regression(
    X_train,
    y_train
)


# 3. Display intercept
print("\nIntercept:")
print(model.intercept_)


# 4. Display coefficients
display_coefficients(
    model,
    X_train.columns
)
print("\nCoefficients:")

# 5. Evaluate model
y_pred = evaluate_model(
    model,
    X_test,
    y_test
)


# 6. Plot results
plot_predictions(
    y_test,
    y_pred
)

# STEP 7 — RESIDUAL ANALYSIS
# ==================================================

residual_summary(
    y_test,
    y_pred
)


# ==================================================
# STEP 8 — VISUALIZATION
# =================================================

plot_residuals(
    y_test,
    y_pred
)
cross_validate(
    model,
    X_train,
    y_train
)
het_breuschpagan(
    model,
    X_test,
    y_test
)