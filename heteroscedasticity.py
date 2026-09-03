import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from multiple_regre import load_and_preprocess_data
from multiple_train import train_linear_regression, display_coefficients
from m_evaluate import evaluate_model, plot_predictions
from diagonistic import plot_residuals, residual_summary
from cross_val import cross_validate

# Dataset path
file_path = r"AIML_T2MODEL\data\Housing.csv"

# 1. Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# 2. Train Linear Regression model
model = train_linear_regression(X_train, y_train)

# 3. Display intercept
print("\nIntercept:")
print(model.intercept_)

# 4. Display coefficients
display_coefficients(model, X_train.columns)

# 5. Evaluate model
y_pred = evaluate_model(model, X_test, y_test)

# 6. Plot results
plot_predictions(y_test, y_pred)

# 7. Residual Analysis
residual_summary(y_test, y_pred)

# 8. Visualization
plot_residuals(y_test, y_pred)

# 9. Cross Validation
cross_validate(model, X_train, y_train)

# 10. Breusch-Pagan Test
residuals = y_test - y_pred
X_test_bp = sm.add_constant(X_test, has_constant="add")
bp_test = het_breuschpagan(residuals, X_test_bp)

print("\n" + "=" * 60)
print("BREUSCH-PAGAN HETEROSCEDASTICITY TEST")
print("=" * 60)
print(f"LM Statistic : {bp_test[0]:.4f}")
print(f"LM p-value   : {bp_test[1]:.4f}")
print(f"F Statistic  : {bp_test[2]:.4f}")
print(f"F p-value    : {bp_test[3]:.4f}")