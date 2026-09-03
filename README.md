# Task3_house_price
House Price Prediction using Simple &amp; Multiple Linear Regression | Data Cleaning, Categorical Encoding, MAE, MSE, R², Coefficient &amp; Residual Analysis with Python and Scikit-learn.
# House Price Prediction using Linear Regression

## 📌 Project Overview

This project implements Simple and Multiple Linear Regression to predict house prices using Python, Pandas, Scikit-learn and Matplotlib.

The project focuses on understanding the complete regression workflow:

- Data cleaning
- Exploratory statistical analysis
- Categorical encoding
- Train-test splitting
- Simple Linear Regression
- Multiple Linear Regression
- Model evaluation
- Regression visualization
- Coefficient interpretation
- Residual analysis
- MUlticolinearity

---

## 🎯 Objective

To implement and understand:

1. Simple Linear Regression
2. Multiple Linear Regression
3. MAE, MSE and R² evaluation metrics
4. Regression coefficients
5. Residual analysis
6. Interpretation of model results

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git
- GitHub

---

## 📊 Dataset

The dataset contains 545 house-property observations.

The target variable is:

`price`

Features include:

- area
- bedrooms
- bathrooms
- stories
- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- parking
- prefarea
- furnishingstatus

---

## 🧹 Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Negative/invalid values
- Categorical inconsistencies
- Numerical distributions
- Potential outliers

No logically invalid negative values were found.

Potential price outliers were identified using the IQR method. 15 high-price observations were flagged and retained because they were considered potentially legitimate properties rather than confirmed data errors.

---

## 🔄 Data Preprocessing

Binary categorical variables were encoded as:

`yes → 1`

`no → 0`

The `furnishingstatus` feature was one-hot encoded.

`drop_first=True` was used so that `furnished` became the reference category.

---

## 📈 Models

### Simple Linear Regression

The relationship between:

`area → price`

was modeled using Simple Linear Regression.

### Multiple Linear Regression

Multiple property characteristics were used simultaneously to predict house price.

General form:

`Price = β0 + β1X1 + β2X2 + ... + βnXn`

---

## 📏 Model Evaluation

Final Multiple Linear Regression results:

| Metric | Result |
|---|---:|
| MAE | 970043.40 |
| MSE | 1754318687330.66 |
| RMSE | ~1324507 |
| R² | 0.6529 |

### Interpretation

The model explains approximately 65.29% of the variation in house prices on the test dataset.

The average absolute prediction error is approximately ₹9.70 lakh.

---

## 🔍 Coefficient Analysis

Selected model coefficients:

- Parking: +₹224,841.91
- Main Road: +₹367,919.95
- Guestroom: +₹231,610.04
- Basement: +₹390,251.18
- Hot Water Heating: +₹684,649.89
- Air Conditioning: +₹791,426.74
- Preferred Area: +₹629,990.57
- Semi-Furnished: −₹126,881.82
- Unfurnished: −₹413,645.06

Coefficients represent conditional associations while holding other included predictors constant.

---

## 📊 Residual Analysis

Mean residual:

`₹146,055.36`

Residual standard deviation:

`₹1,322,510`

Residual analysis was used to understand the distribution and magnitude of prediction errors.

---

## 💡 Key Learnings

- How to clean numerical and categorical data
- How to identify potential outliers using IQR
- How to encode categorical variables
- Why one dummy category is dropped
- How Simple Linear Regression works
- How Multiple Linear Regression extends Simple Linear Regression
- How MAE, MSE and R² measure model performance
- How to interpret regression coefficients
- Why "holding other variables constant" is important
- How residuals help diagnose model errors

---

## 🚀 Future Improvements

- Compare Linear Regression with Ridge and Lasso Regression
- Perform residual diagnostic plots
- Check multicollinearity using VIF
- Apply feature scaling where appropriate
- Compare additional regression algorithms
- Perform cross-validation
- Build an interactive Streamlit prediction dashboard

---

## 👨‍💻 Author
Nikhil Bhoyar
Data Science / Machine Learning Project
